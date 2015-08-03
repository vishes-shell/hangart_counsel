from django.db import models

from ..users.models import User
from ..shop.models import Picture


class Rating(models.Model):
    picture = models.ForeignKey(Picture)
    rating = models.PositiveSmallIntegerField(max_length=1)
    user = models.ForeignKey(User, null=True, blank=True)
    session_key = models.CharField(max_length=40, blank=True)
    active_user = models.BooleanField(default=True)

    def __str__(self):
        if self.user:
            return '[Rating:{0}] [User:{1}]'.format(self.rating, self.user)
        else:
            return '[Rating:{0}] [Session:{1}]'.format(self.rating, self.session_key)


class RatingPerson(models.Model):
    email = models.EmailField()
    user = models.ForeignKey(User, blank=True, null=True)
    session_key = models.CharField(max_length=40, blank=True)

    def __str__(self):
        if self.user:
            return '[Email: {0}] [User:{1}]'.format(self.email, self.user)
        else:
            return '[Email: {0}] [Session:{1}]'.format(self.email, self.session_key)