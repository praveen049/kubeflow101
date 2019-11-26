from typing import NamedTuple

import kfp
from kfp.components import InputPath, InputTextFile, InputBinaryFile, OutputPath, OutputTextFile, OutputBinaryFile
from kfp.components import func_to_container_op
import kfp.compiler as compiler

# Writing bigger data
@func_to_container_op
def repeat_line(line: str, output_text_path: OutputPath(str), count: int = 10):
    '''Repeat the line specified number of times'''
    with open(output_text_path, 'w') as writer:
        for i in range(count):
            writer.write(line + '\n')


# Reading bigger data
@func_to_container_op
def print_text(text_path: InputPath(str)):
    '''Print text'''
    with open(text_path, 'r') as reader:
        for line in reader:
            print(line, end = '')

def print_repeating_lines_pipeline():
    print_text(repeat_line(line='Hello', count=5).output) # Don't forget .output !

if __name__ == '__main__':
    compiler.Compiler().compile(print_repeating_lines_pipeline, 'writing-data-indirectly-1.zip')
