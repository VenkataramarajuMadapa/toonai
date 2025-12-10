from django.db import models

class Cartoon(models.Model):
    tvmaze_id = models.IntegerField(unique=True)

    name = models.CharField(max_length=255)
    summary = models.TextField(blank=True, null=True)
    genres = models.CharField(max_length=500, blank=True, null=True)

    rating = models.FloatField(blank=True, null=True)
    premiered = models.CharField(max_length=20, blank=True, null=True)

    image_medium = models.URLField(max_length=500, blank=True, null=True)
    image_original = models.URLField(max_length=500, blank=True, null=True)

    # Extra useful fields (future-proof)
    status = models.CharField(max_length=50, blank=True, null=True)
    language = models.CharField(max_length=50, blank=True, null=True)
    runtime = models.IntegerField(blank=True, null=True)
    official_site = models.URLField(max_length=500, blank=True, null=True)
    updated = models.IntegerField(blank=True, null=True)  # tvmaze timestamp
    popularity = models.IntegerField(blank=True, null=True)  # fallback ranking

    def __str__(self):
        return self.name
