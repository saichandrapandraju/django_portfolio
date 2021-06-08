from django.db import models

# Create your models here.

class Image_data(models.Model):
    image = models.FileField(upload_to='projects/uploaded_images')