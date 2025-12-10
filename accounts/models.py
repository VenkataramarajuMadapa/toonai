from django.db import models
from django.conf import settings
from django.contrib.auth.models import AbstractUser

class User(AbstractUser):
    email = models.EmailField(unique=True)
    phone = models.CharField(max_length=20, unique=True)
    profile_picture = models.ImageField(upload_to='profile_pictures/', null=True, blank=True)
    address = models.CharField(max_length=150, blank=True, null=True)
    registered_on = models.DateTimeField(auto_now_add=True)
    last_active = models.DateTimeField(null=True, blank=True)

    REQUIRED_FIELDS = ['email', 'phone', 'first_name', 'last_name']
    USERNAME_FIELD = 'username'

    def get_profile_picture(self):
        if self.profile_picture:
            return self.profile_picture.url
        return f"{settings.STATIC_URL}images/default-avatar.png"
