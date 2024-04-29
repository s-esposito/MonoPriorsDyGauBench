import sys

def replace_content(input_file, output_file):
    with open(input_file, 'r') as file:
        content = file.read()
    
    content = content.replace('hypernerf', 'dnerf')
    content = content.replace('trex', 'lego')
    
    with open(output_file, 'w') as file:
        file.write(content)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python replace_content.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    replace_content(input_file, output_file)