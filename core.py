from colorama import Fore, Style, init, just_fix_windows_console
import math
import sys
import os
import importlib.util

# Initialize colorama with specific settings for ComfyUI
# This ensures colors work even when stdout/stderr are redirected
init(autoreset=True, convert=True, strip=False, wrap=True)
just_fix_windows_console()  # Fix Windows console issues

COMPARE_FUNCTIONS = {
    "a == b": lambda a, b: a == b,
    "a != b": lambda a, b: a != b,
    "a < b": lambda a, b: a < b,
    "a > b": lambda a, b: a > b,
    "a <= b": lambda a, b: a <= b,
    "a >= b": lambda a, b: a >= b,
}

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

any_type = AnyType("*")

NONE = "None"

MODEL_DIFFUSION = "Diffusion"
MODEL_GGUF = "GGUF"
MODEL_TYPE_LIST = [MODEL_DIFFUSION, MODEL_GGUF]

CLIP_STABLE_DIFFUSION = "stable_diffusion"
CLIP_STABLE_CASCADE = "stable_cascade"
CLIP_SD3 = "sd3"
CLIP_STABLE_AUDIO = "stable_audio"
CLIP_MOCHI = "mochi"
CLIP_LTXV = "ltxv"
CLIP_PIXART = "pixart"
CLIP_COSMOS = "cosmos"
CLIP_LUMINA2 = "lumina2"
CLIP_WAN = "wan"
CLIP_HIDREAM = "hidream"
CLIP_CHROMA = "chroma"
CLIP_ACE = "ace"
CLIP_OMNIGEN2 = "omnigen2"
CLIP_TYPE_LIST = [CLIP_STABLE_DIFFUSION, CLIP_STABLE_CASCADE, CLIP_SD3, CLIP_STABLE_AUDIO, CLIP_MOCHI, CLIP_LTXV, CLIP_PIXART, CLIP_COSMOS, CLIP_LUMINA2, CLIP_WAN, CLIP_HIDREAM, CLIP_CHROMA, CLIP_ACE, CLIP_OMNIGEN2]

CLIP_DEVICE_DEFAULT = "default"
CLIP_DEVICE_CPU = "cpu"
CLIP_DEVICE_LIST = [CLIP_DEVICE_DEFAULT, CLIP_DEVICE_CPU]

WAN_480P = 'Wan 480p'
WAN_720P = 'Wan 720p'
WAN_MODELS = [WAN_480P, WAN_720P]
WAN_MODELS_CONFIG = {
    WAN_480P: { 'max_side': 832, 'max_pixels': 832 * 480, 'model_name': WAN_480P},
    WAN_720P: { 'max_side': 1280, 'max_pixels': 1280 * 720, 'model_name': WAN_720P},
}

START_IMAGE = "Start Image"
END_IMAGE = "End Image"
START_END_IMAGE = "Start to End Image"
END_TO_START_IMAGE = "End to Start Image"
START_TO_END_TO_START_IMAGE = "Start to End to Start Image"
WAN_FIRST_END_FIRST_FRAME_TP_VIDEO_MODE = [START_IMAGE, END_IMAGE, START_END_IMAGE, END_TO_START_IMAGE, START_TO_END_TO_START_IMAGE]

BLACK = "\033[90m"  # Bright black (gray)
RED = "\033[91m"    # Bright red
GREEN = "\033[92m"  # Bright green
YELLOW = "\033[93m" # Bright yellow
BLUE = "\033[94m"   # Bright blue
MAGENTA = "\033[95m"# Bright magenta
CYAN = "\033[96m"   # Bright cyan
WHITE = "\033[97m"  # Bright white
RESET = "\033[0m"


def _supports_color():
    """Check if the current environment supports ANSI colors"""
    # Check common environment variables
    if os.environ.get('NO_COLOR'):
        return False
    if os.environ.get('FORCE_COLOR'):
        return True
    
    # Check if we're in a terminal
    if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
        return True
    
    # Check for common color-supporting terminals
    term = os.environ.get('TERM', '').lower()
    if any(x in term for x in ['color', 'ansi', 'xterm', 'screen', 'tmux']):
        return True
    
    # On Windows, assume colors work after colorama initialization
    if sys.platform == 'win32':
        return True
    
    # Special case: ComfyUI's LogInterceptor - check if underlying stream supports color
    if hasattr(sys.stdout, 'buffer') and hasattr(sys.stdout.buffer, 'isatty'):
        try:
            return sys.stdout.buffer.isatty()
        except:
            pass
    
    return False


def _get_comfyui_stream():
    """Try to get the original ComfyUI stream for better color support"""
    # Check if we're dealing with ComfyUI's LogInterceptor
    if hasattr(sys.stdout, '__class__') and 'LogInterceptor' in sys.stdout.__class__.__name__:
        # Try to get the underlying stream
        if hasattr(sys.stdout, 'buffer'):
            return sys.stdout.buffer
    return sys.stdout


def _colorize_with_fallback(text: str, color_code: str, colorama_color=None):
    """Apply color with fallback strategies for different environments"""
    if not _supports_color():
        return text
    
    # Strategy 1: Try colorama first (works well with ComfyUI's LogInterceptor)
    if colorama_color:
        try:
            # Use the brightest colorama variant if available
            colorama_map = {
                Fore.BLUE: Fore.LIGHTBLUE_EX,
                Fore.CYAN: Fore.LIGHTCYAN_EX,
                Fore.GREEN: Fore.LIGHTGREEN_EX,
                Fore.RED: Fore.LIGHTRED_EX,
                Fore.MAGENTA: Fore.LIGHTMAGENTA_EX,
                Fore.YELLOW: Fore.LIGHTYELLOW_EX,
                Fore.WHITE: Fore.LIGHTWHITE_EX,
            }
            bright_color = colorama_map.get(colorama_color, colorama_color)
            return f"{bright_color}{text}{Style.RESET_ALL}"
        except:
            pass
    
    # Strategy 2: Fall back to ANSI codes
    return f"{color_code}{text}{RESET}"


def _print_with_color_support(*args, **kwargs):
    """Enhanced print function that works better with ComfyUI"""
    # Use the standard print, but ensure colorama processes it correctly
    try:
        # Force colorama to process the output
        print(*args, **kwargs)
        # Ensure immediate flush for real-time output
        sys.stdout.flush()
    except Exception as e:
        # Fallback to basic print if anything goes wrong
        print(*args, **kwargs)


def _print_direct_to_console(text: str):
    """Alternative approach: try to write directly to console when possible"""
    try:
        # Strategy 1: Try to get the original console on Windows
        if sys.platform == 'win32':
            try:
                import ctypes
                from ctypes import wintypes
                
                # Get console handle
                STD_OUTPUT_HANDLE = -11
                console_handle = ctypes.windll.kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
                
                if console_handle and console_handle != -1:
                    # Try to write directly to console
                    text_bytes = text.encode('utf-8')
                    chars_written = wintypes.DWORD()
                    ctypes.windll.kernel32.WriteConsoleA(
                        console_handle, 
                        text_bytes, 
                        len(text_bytes), 
                        ctypes.byref(chars_written), 
                        None
                    )
                    return True
            except:
                pass
        
        # Strategy 2: Try to access underlying streams
        if hasattr(sys.stdout, 'buffer'):
            try:
                sys.stdout.buffer.write(text.encode('utf-8'))
                sys.stdout.buffer.flush()
                return True
            except:
                pass
        
        # Strategy 3: Standard approach
        print(text, end='')
        sys.stdout.flush()
        return True
        
    except:
        return False


def output_to_terminal(text: str):
    colored_text = _colorize_with_fallback(text, CYAN, Fore.CYAN)
    prefix = _colorize_with_fallback("[Przewodo UTILS]", BLUE, Fore.BLUE)
    message = f"{RESET}{prefix} {colored_text}\n"
    # Try multiple output strategies
    if not _print_direct_to_console(message):
        _print_with_color_support(f"{RESET}{prefix} {colored_text}")

def output_to_terminal_successful(text: str):
    colored_text = _colorize_with_fallback(text, GREEN, Fore.GREEN)
    prefix = _colorize_with_fallback("[Przewodo UTILS]", BLUE, Fore.BLUE)
    message = f"{RESET}{prefix} {colored_text}\n"
    # Try multiple output strategies
    if not _print_direct_to_console(message):
        _print_with_color_support(f"{RESET}{prefix} {colored_text}")

def output_to_terminal_error(text: str):
    colored_text = _colorize_with_fallback(text, RED, Fore.RED)
    prefix = _colorize_with_fallback("[Przewodo UTILS]", BLUE, Fore.BLUE)
    message = f"{RESET}{prefix} {colored_text}\n"
    # Try multiple output strategies
    if not _print_direct_to_console(message):
        _print_with_color_support(f"{RESET}{prefix} {colored_text}")

def debug_color_support():
    """Debug function to check color support and environment"""
    info = []
    info.append(f"Python platform: {sys.platform}")
    info.append(f"stdout type: {type(sys.stdout).__name__}")
    info.append(f"stderr type: {type(sys.stderr).__name__}")
    info.append(f"stdout isatty: {getattr(sys.stdout, 'isatty', lambda: False)()}")
    info.append(f"Color support detected: {_supports_color()}")
    info.append(f"TERM: {os.environ.get('TERM', 'not set')}")
    info.append(f"NO_COLOR: {os.environ.get('NO_COLOR', 'not set')}")
    info.append(f"FORCE_COLOR: {os.environ.get('FORCE_COLOR', 'not set')}")
    
    # Check if we're in ComfyUI's LogInterceptor
    if hasattr(sys.stdout, '__class__') and 'LogInterceptor' in sys.stdout.__class__.__name__:
        info.append("ComfyUI LogInterceptor detected")
        if hasattr(sys.stdout, 'buffer'):
            info.append(f"Underlying buffer type: {type(sys.stdout.buffer).__name__}")
    
    for line in info:
        print(f"[Przewodo UTILS DEBUG] {line}")


def test_all_colors():
    """Test function to verify all color outputs work correctly"""
    print("\n" + "="*60)
    print("Testing Przewodo UTILS Color Output Functions")
    print("="*60)
    
    # Test basic colors first
    print("\n1. Testing basic color support:")
    debug_color_support()
    
    print("\n2. Testing direct ANSI codes:")
    print(f"{BLUE}[DIRECT ANSI]{RESET} {CYAN}This should be cyan{RESET}")
    print(f"{BLUE}[DIRECT ANSI]{RESET} {GREEN}This should be green{RESET}")
    print(f"{BLUE}[DIRECT ANSI]{RESET} {RED}This should be red{RESET}")
    
    print("\n3. Testing colorama approach:")
    print(f"{Fore.LIGHTBLUE_EX}[COLORAMA]{Style.RESET_ALL} {Fore.LIGHTCYAN_EX}This should be cyan{Style.RESET_ALL}")
    print(f"{Fore.LIGHTBLUE_EX}[COLORAMA]{Style.RESET_ALL} {Fore.LIGHTGREEN_EX}This should be green{Style.RESET_ALL}")
    print(f"{Fore.LIGHTBLUE_EX}[COLORAMA]{Style.RESET_ALL} {Fore.LIGHTRED_EX}This should be red{Style.RESET_ALL}")
    
    print("\n4. Testing Przewodo UTILS functions:")
    output_to_terminal("This is a regular message")
    output_to_terminal_successful("This is a success message") 
    output_to_terminal_error("This is an error message")
    
    print("\n5. Testing fallback colorization:")
    print(_colorize_with_fallback("Fallback cyan test", CYAN, Fore.CYAN))
    print(_colorize_with_fallback("Fallback green test", GREEN, Fore.GREEN))
    print(_colorize_with_fallback("Fallback red test", RED, Fore.RED))
    
    print("\n" + "="*60)
    print("Color test complete!")
    print("If you see colors above, the functions are working correctly.")
    print("If not, try setting FORCE_COLOR=1 environment variable.")
    print("="*60 + "\n")


def import_nodes(paths, nodes):
    """
    Import custom nodes dynamically from ComfyUI custom_nodes directory.
    
    Args:
        paths (list): Array of strings containing the path components for the custom node
                     Example: ["teacache"] or ["comfyui-kjnodes", "nodes"] 
        nodes (list): Array of strings containing the class names to import
                     Example: ["TeaCache"] or ["SkipLayerGuidanceWanVideo", "UnetLoaderGGUF"]
    
    Returns:
        dict: Dictionary mapping node class names to the imported classes (or None if failed)
              Example: {"TeaCache": <class>, "SkipLayerGuidanceWanVideo": None}
    """
    result = {}
    
    try:
        # Build the path to the custom node
        custom_nodes_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "custom_nodes")
        node_path = os.path.join(custom_nodes_path, *paths)
        
        # Determine the module file to import
        module_file = None
        package_name = None
        is_package = False
        
        # Strategy 1: If the last path component is a .py file
        if paths[-1].endswith('.py'):
            module_file = node_path
            package_name = ".".join(paths[:-1]) if len(paths) > 1 else None
        # Strategy 2: If it's a directory, look for nodes.py
        elif os.path.isdir(node_path):
            module_file = os.path.join(node_path, "nodes.py")
            package_name = ".".join(paths)
            is_package = True
        # Strategy 3: If it's a path component that should be a file
        else:
            # Try adding .py extension
            module_file = node_path + ".py"
            if not os.path.exists(module_file):
                # Try looking for nodes.py in the directory
                if os.path.isdir(os.path.dirname(node_path)):
                    module_file = os.path.join(os.path.dirname(node_path), "nodes.py")
                    package_name = ".".join(paths[:-1]) if len(paths) > 1 else None
        
        if not module_file or not os.path.exists(module_file):
            # Try alternative locations for common custom node structures
            alternatives = []
            base_path = os.path.join(custom_nodes_path, paths[0])
            
            if os.path.isdir(base_path):
                # Common file names to try
                common_files = ["nodes.py", "__init__.py", f"{paths[0]}.py"]
                for filename in common_files:
                    alt_file = os.path.join(base_path, filename)
                    if os.path.exists(alt_file):
                        alternatives.append(alt_file)
                        package_name = paths[0]
                        is_package = True
                        break
            
            if alternatives:
                module_file = alternatives[0]
            else:
                raise ImportError(f"Module file not found. Tried: {module_file}, alternatives in {base_path}")
        
        # Create a unique module name to avoid conflicts
        module_name = "_".join(paths).replace("-", "_").replace("/", "_").replace("", "_") + "_import"
        
        # For packages with relative imports, we need to set up the package structure
        if is_package and package_name:
            # First, make sure the package directory is in sys.path temporarily
            package_dir = os.path.dirname(module_file)
            if package_dir not in sys.path:
                sys.path.insert(0, package_dir)
                path_added = True
            else:
                path_added = False
            
            try:
                # Import the package as a module with proper name
                package_module_name = package_name.replace("-", "_")
                
                # Load the module with the package context
                spec = importlib.util.spec_from_file_location(package_module_name, module_file, submodule_search_locations=[])
                if spec is None:
                    raise ImportError(f"Could not create module spec for {module_file}")
                
                module = importlib.util.module_from_spec(spec)
                if module is None:
                    raise ImportError(f"Could not create module from spec for {module_file}")
                
                # Set up the module as a package if it has relative imports
                module.__package__ = package_module_name
                module.__path__ = [package_dir]
                
                # Add the module to sys.modules before execution to handle relative imports
                sys.modules[package_module_name] = module
                
                try:
                    spec.loader.exec_module(module)
                except Exception as e:
                    # Clean up sys.modules on failure
                    if package_module_name in sys.modules:
                        del sys.modules[package_module_name]
                    raise ImportError(f"Failed to execute module {module_file}: {e}")
                
            finally:
                # Clean up sys.path
                if path_added and package_dir in sys.path:
                    sys.path.remove(package_dir)
        else:
            # Simple module import (no relative imports expected)
            spec = importlib.util.spec_from_file_location(module_name, module_file)
            if spec is None:
                raise ImportError(f"Could not create module spec for {module_file}")
            
            module = importlib.util.module_from_spec(spec)
            if module is None:
                raise ImportError(f"Could not create module from spec for {module_file}")
            
            # Add the module to sys.modules temporarily
            sys.modules[module_name] = module
            
            try:
                spec.loader.exec_module(module)
            except Exception as e:
                # Clean up sys.modules on failure
                if module_name in sys.modules:
                    del sys.modules[module_name]
                raise ImportError(f"Failed to execute module {module_file}: {e}")
        
        # Import the requested node classes
        success_count = 0
        for node_class in nodes:
            if hasattr(module, node_class):
                result[node_class] = getattr(module, node_class)
                output_to_terminal_successful(f"{node_class} imported successfully from {'/'.join(paths)}!")
                success_count += 1
            else:
                result[node_class] = None
                output_to_terminal_error(f"Warning: {node_class} not found in {'/'.join(paths)}")
        
        # Clean up sys.modules after successful import (but keep the package if it's needed)
        if not is_package and module_name in sys.modules:
            del sys.modules[module_name]
            
    except (ImportError, AttributeError, OSError) as e:
        # If import fails, set all nodes to None
        for node_class in nodes:
            result[node_class] = None
        output_to_terminal_error(f"Warning: Failed to import from {'/'.join(paths)} ({e}). Please check installation.")
    
    return result
# Re-initialize colorama at the end to ensure proper setup
init(autoreset=True, convert=True, strip=False)