from django.urls import path
from . import views

urlpatterns = [
    path('update-status/<int:cartoon_id>/<str:status>/', views.update_user_status, name='update_user_status'),
    path('my-list/', views.user_watchlist, name='user_watchlist'),
    path("recommendations/", views.recommendations_page, name="recommendations"),
]
