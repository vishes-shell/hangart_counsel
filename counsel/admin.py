from django.contrib import admin

from .models import *


models_to_register = [Rating, RatingPerson]

admin.site.register(models_to_register)