import kfp.dsl as dsl
import kfp.compiler as compiler

@dsl.pipeline(name='load-predict-pipeline-logreg')
def pipeline(project_id='loan-predict'):
    preprocessor = dsl.ContainerOp(
        name='preprocessor',
        image='praveen049/loan-predict-logreg-preproc',
        command=['python', 'preprocessor.py'],
        arguments=[
            '--output-x', '/x.pkl',
            '--output-x-path-file', '/x.txt',
            '--output-y', '/y.pkl',
            '--output-y-path-file', '/y.txt',
    ],
        file_outputs={
            'x-output': '/x.pkl',
            'y-output': '/y.pkl',
        }
    )
    trainer = dsl.ContainerOp(
        name='trainer',
        image='praveen049/loan-predict-logreg-train',
        command=['python', 'train.py'],
        arguments=[
            '--input_x_path_file', dsl.InputArgumentPath(preprocessor.outputs['x-output']),
            '--input_y_path_file', dsl.InputArgumentPath(preprocessor.outputs['y-output']),
            '--output_model', '/model.pkl',
            '--output_model_path_file', '/model.txt',
        ],
        file_outputs={
            'model': '/model.pkl',
        }
    )
    trainer.after(preprocessor)


if __name__ == '__main__':
    compiler.Compiler().compile(pipeline, 'load-predict-pipeline-logreg.zip')
