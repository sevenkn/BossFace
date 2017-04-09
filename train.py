import os
import os.path
from clarifai import rest
from clarifai.rest import ClarifaiApp
from clarifai.rest import Image as ClImage


app = ClarifaiApp()
filepath = "C:\Users\hacker\Downloads\doss"
for f in os.listdir(filepath):
    filename = os.path.join('%s\%s' % (filepath,f))
    app.inputs.create_image_from_filename(filename,concepts=['fangfang'])
app.models.delete_all()
model = app.models.create(model_id="boss",concepts=['fangfang'])
#model = app.models.get('boss')
model = model.train()

image = ClImage(file_obj=open('boss.jpg','rb'))
probability = model.predict([image])['outputs'][0]['data']['concepts'][0]['value']
print probability
