# Math OCR Project Startup Script
# This script activates virtual environment and runs project commands

PROJECT_NAME="Math OCR Project"
VENV_PATH="venv"

echo "🚀 Starting $PROJECT_NAME..."
echo "======================================"

# Function to check if we're in the right directory
check_project_dir() {
    if [[ ! -f "requirements.txt" || ! -d "src" ]]; then
        echo "❌ Error: Not in project root directory!"
        echo "   Please run this script from the math-eqn-ocr/ directory"
        echo "   Current directory: $(pwd)"
        exit 1
    fi
}

# Function to create virtual environment if it doesn't exist
setup_venv() {
    if [[ ! -d "$VENV_PATH" ]]; then
        echo "📦 Virtual environment not found. Creating one..."
        python3 -m venv $VENV_PATH
        if [[ $? -eq 0 ]]; then
            echo "✅ Virtual environment created successfully"
        else
            echo "❌ Failed to create virtual environment"
            exit 1
        fi
    else
        echo "✅ Virtual environment found"
    fi
}

# Function to activate virtual environment
activate_venv() {
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        # Windows
        source $VENV_PATH/Scripts/activate
    else
        # Linux/Mac
        source $VENV_PATH/bin/activate
    fi
    
    if [[ $? -eq 0 ]]; then
        echo "✅ Virtual environment activated"
        echo "   Python: $(which python)"
        echo "   Pip: $(which pip)"
    else
        echo "❌ Failed to activate virtual environment"
        exit 1
    fi
}

# Function to install/update requirements
install_requirements() {
    echo "📦 Installing/updating requirements..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    if [[ $? -eq 0 ]]; then
        echo "✅ Requirements installed successfully"
    else
        echo "❌ Failed to install requirements"
        exit 1
    fi
}

# Function to test the setup
test_setup() {
    echo "🧪 Testing project setup..."
    
    echo "  📝 Testing vocabulary..."
    python src/data/vocabulary.py
    if [[ $? -ne 0 ]]; then
        echo "❌ Vocabulary test failed"
        return 1
    fi
    
    echo "  📊 Testing dataloader..."
    python -c "from src.data.dataloader import test_dataloader; test_dataloader()"
    if [[ $? -ne 0 ]]; then
        echo "❌ DataLoader test failed"
        return 1
    fi
    
    echo "✅ All tests passed!"
    return 0
}

# Function to show project status
show_status() {
    echo ""
    echo "📊 PROJECT STATUS:"
    echo "=================="
    echo "📁 Project directory: $(pwd)"
    echo "🐍 Python version: $(python --version)"
    echo "📦 Pip version: $(pip --version)"
    echo "📚 Installed packages:"
    pip list | grep -E "(torch|numpy|opencv|transformers|albumentations)" | head -5
    echo ""
    echo "📊 Dataset status:"
    if [[ -d "data/raw/crohme" ]]; then
        inkml_count=$(find data/raw/crohme -name "*.inkml" | wc -l)
        echo "   ✅ CROHME dataset found ($inkml_count InkML files)"
    else
        echo "   ❌ CROHME dataset not found"
    fi
    echo ""
}

# Function to provide next steps
show_next_steps() {
    echo "🎯 NEXT STEPS:"
    echo "=============="
    echo "1. 📊 Explore the data:"
    echo "   python -c \"from src.data.dataloader import test_dataloader; test_dataloader()\""
    echo ""
    echo "2. 🏗️ Create the model (when ready):"
    echo "   python src/models/math_ocr_model.py"
    echo ""
    echo "3. 🏋️ Start training (when ready):"
    echo "   python src/main.py"
    echo ""
    echo "4. 📊 Monitor with Jupyter:"
    echo "   jupyter notebook notebooks/"
    echo ""
    echo "💡 To deactivate virtual environment later, just run: deactivate"
}

# Function to handle different startup modes
handle_startup_mode() {
    case "$1" in
        "--quick" | "-q")
            echo "⚡ Quick start mode (skip tests)"
            return 0
            ;;
        "--test-only" | "-t")
            echo "🧪 Test-only mode"
            setup_venv
            activate_venv
            test_setup
            exit $?
            ;;
        "--install-only" | "-i")
            echo "📦 Install-only mode"
            return 0
            ;;
        "--help" | "-h")
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -q, --quick       Quick start (skip tests)"
            echo "  -t, --test-only   Run tests only"
            echo "  -i, --install-only Install requirements only"
            echo "  -h, --help        Show this help"
            echo ""
            exit 0
            ;;
        *)
            return 0
            ;;
    esac
}

# Main execution
main() {
    # Handle command line arguments
    handle_startup_mode "$1"
    startup_mode=$?
    
    # Core setup steps
    check_project_dir
    setup_venv
    activate_venv
    
    # Install requirements unless quick mode
    if [[ "$1" != "--test-only" ]]; then
        install_requirements
    fi
    
    # Run tests unless install-only or quick mode
    if [[ "$1" != "--install-only" && "$1" != "--quick" ]]; then
        test_setup
        test_status=$?
    else
        test_status=0
    fi
    
    # Show status and next steps
    show_status
    
    if [[ $test_status -eq 0 ]]; then
        show_next_steps
        echo ""
        echo "🎉 Project ready! You're now in the activated virtual environment."
        echo "   Run 'deactivate' to exit the environment when done."
    else
        echo "⚠️  Setup completed with some test failures."
        echo "   Check the error messages above and fix any issues."
    fi
}

# Run main function with all arguments
main "$@"