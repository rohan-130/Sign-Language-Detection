"""asl_recognition URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
import camera.views


urlpatterns = [
    path('admin/', admin.site.urls),
    path('test-cam/', camera.views.test_cam),
    path('train-cam/', camera.views.train_cam),
    path('get-words/', camera.views.get_all_words),
    path('check-word/', camera.views.check_word),
    path('train-word/', camera.views.train_word),
]
