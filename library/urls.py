from django.urls import path
from . import views

urlpatterns = [
    path("cartoons/", views.cartoon_list, name="cartoon_list"),
    path("cartoon/<int:cartoon_id>/", views.cartoon_detail, name="cartoon_detail"),
    path("rate/<int:cartoon_id>/", views.rate_cartoon, name="rate_cartoon"),
    path("review/<int:cartoon_id>/", views.review_cartoon, name="review_cartoon"),
    path("genres/", views.genre_list, name="genre_list"),
    path("genres/<str:genre_name>/", views.genre_detail, name="genre_detail"),
]
