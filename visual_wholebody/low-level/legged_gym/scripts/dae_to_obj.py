import pyassimp
import os
import sys

def convert_dae_to_obj(input_file, output_file):
    try:
        # Load the .dae file
        scene = pyassimp.load(input_file)
        print(f"Successfully loaded {input_file}")

        # Export the scene to .obj format
        if not pyassimp.export(scene, output_file, format='obj'):
            raise Exception("Export failed!")

        print(f"File successfully converted and saved as {output_file}")

    except Exception as e:
        print(f"Error during conversion: {e}")

    finally:
        # Always release the scene to free memory
        pyassimp.release(scene)

if __name__ == "__main__":
    # Example usage: python script.py input.dae output.obj
    if len(sys.argv) < 3:
        print("Usage: python convert_dae_to_obj.py input_file.dae output_file.obj")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        sys.exit(1)

    # Perform conversion
    convert_dae_to_obj(input_file, output_file)
