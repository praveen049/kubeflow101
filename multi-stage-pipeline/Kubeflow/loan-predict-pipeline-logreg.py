import kfp.dsl as dsl
import kfp.compiler as compiler

@dsl.pipeline(name='load-predict-pipeline-logreg')
def pipeline(project_id='loan-predict'):
    preprocessor = dsl.ContainerOp(
        name='preprocessor',
        image='praveen049/loan-predict-logreg-preproc',
        command=['python', 'preprocessor.py'],
        arguments=[
            '--output-x', '/tmp/x.pkl',
            '--output-x-path-file', '/tmp/x.txt',
            '--output-y', '/tmp/y.pkl',
            '--output-y-path-file', '/tmp/y.txt',
    ],
        file_outputs={
            'x-output': '/tmp/x.txt',
            'y-output': '/tmp/y.txt',
        }
    )
    trainer = dsl.ContainerOp(
        name='trainer',
        image='praveen049/loan-predict-logreg-train',
        command=['python', 'train.py'],
        arguments=[
            '--input_x_path_file', preprocessor.outputs['x-output'],
            '--input_y_path_file', preprocessor.outputs['y-output'],
            '--output_model', '/tmp/model.pkl',
            '--output_model_path_file', '/tmp/model.txt'
        ],
        file_outputs={
            'model': '/tmp/model.pkl',
            'model-path': '/tmp/model.txt',
        }
    )
    trainer.after(preprocessor)


if __name__ == '__main__':
    compiler.Compiler().compile(pipeline, 'load-predict-pipeline-logreg.zip')
