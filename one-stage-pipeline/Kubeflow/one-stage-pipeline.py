import kfp
from kfp import compiler
import kfp.compiler as compiler
import kfp.components as comp
import kfp.dsl as dsl

@kfp.dsl.component
def my_component():
  return kfp.dsl.ContainerOp(
    name='infer',
    image='praveen049/inf'
  )

@kfp.dsl.pipeline(
  name='pipe1',
  description='My machine learning pipeline'
)
def my_pipeline():
  one_step = my_component()

kfp.compiler.Compiler().compile(my_pipeline,'my-pipeline.zip')
