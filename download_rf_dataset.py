from roboflow import Roboflow


rf = Roboflow(api_key="t9t1f7lIoBrbIhkqHsmH")
project = rf.workspace("semanticsegmentation-8o5vb").project("semanticsegmentation_dataset12052023")
dataset = project.version(17).download("png-mask-semantic")
