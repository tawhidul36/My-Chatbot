from django.urls import path
from . import views

urlpatterns = [
    path("", views.chatbot_view, name="chatbot"),
    path("ask/", views.get_answer, name="ask_bot"),
]