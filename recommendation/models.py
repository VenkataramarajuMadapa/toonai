from django.db import models
from django.conf import settings
from django.utils import timezone
from library.models import Cartoon

class UserRating(models.Model):

    STATUS_CHOICES = [
        ('watchlist', 'Watchlist'),
        ('watching', 'Watching'),
        ('completed', 'Completed'),
        ('canceled', 'Canceled'),
        ('not_interested', 'Not Interested'),
    ]

    user = models.ForeignKey(settings.AUTH_USER_MODEL,on_delete=models.CASCADE)
    cartoon = models.ForeignKey(Cartoon, on_delete=models.CASCADE)
    status = models.CharField(max_length=20,choices=STATUS_CHOICES,default='watchlist')
    rating = models.FloatField(null=True, blank=True)
    review = models.TextField(blank=True, null=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'cartoon')

    def __str__(self):
        return f"{self.user.username} → {self.cartoon.name} ({self.status})"