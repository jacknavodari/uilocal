import os
import sys
import json
import logging

# Set up logginglogging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('pyi_rthook')

# Get the path to the bundled files
def _get_bundle_dir():
    if getattr(sys, 'frozen', False):
        # we are running in a bundle
        return sys._MEIPASS
    else:
        # we are running in a normal Python environment
        return os.path.dirname(os.path.abspath(__file__))

# Add the bundle directory to the system path
sys.path.append(_get_bundle_dir())

# Fix for Gradio client files
try:
    from pathlib import Path
    
    # Get the path to the bundled files
    bundle_dir = Path(_get_bundle_dir())
    
    # Create a symlink to the gradio_client directory if it exists in the bundle
    internal_dir = bundle_dir / "_internal"
    if internal_dir.exists():
        gradio_client_path = internal_dir / "gradio_client"
        if gradio_client_path.exists():
            # Add the gradio_client directory to the Python path
            sys.path.insert(0, str(gradio_client_path))
            logger.info(f"Added gradio_client path: {gradio_client_path}")

            # Create a types.json file in the expected location if it doesn't exist
            types_json_path = gradio_client_path / "types.json"
            if not types_json_path.exists():
                # Create a minimal types.json file
                with open(types_json_path, 'w') as f:
                    json.dump({"types": {}}, f)
                logger.info(f"Created dummy types.json at: {types_json_path}")

except Exception as e:
    logger.error(f"Error in runtime hook: {str(e)}", exc_info=True)