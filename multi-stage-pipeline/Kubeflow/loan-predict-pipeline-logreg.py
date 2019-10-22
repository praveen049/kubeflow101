import kfp.dsl as dsl
import kfp.compiler as compiler

@dsl.pipeline(name='load-predict-pipeline-logreg')
def pipeline(project_id='loan-predict'):
    preprocessor = dsl.ContainerOp(
        name='preprocessor',
        image='praveen049/loan-predict-logreg-preproc',
        command=['python', 'preproc/preprocessor.py'],
        arguments=[
            '--project_id', project_id,
            '--output-x-path', 'x.pkl',
            '--output-x-path-file', 'x.txt',
            '--output-y-path', 'y.pkl',
            '--output-y-path-file', 'y.txt',
    ],
        file_outputs={
            'x-output': 'x.pkl',
            'y-output': 'y.pkl',
        }
    )
    trainer = dsl.ContainerOp(
        name='trainer',
        image='praveen049/loan-predict-logreg-train',
        command=['python', 'train/train.py'],
        arguments=[
            '--project_id', project_id,
            '--input-x-path', preprocessor.outputs['x-output'],
            '--input-y-path', preprocessor.outputs['y-output'],
            '--output-model-path', 'model.pkl',
            '--output-model-path-file', 'model.txt'
        ],
        file_outputs={
            'model': 'model.pkl',
            'model-path': 'model.txt',
        }
    )
    trainer.after(preprocessor)


if __name__ == '__main__':
    compiler.Compiler().compile(pipeline, 'load-predict-pipeline-logreg.zip')
