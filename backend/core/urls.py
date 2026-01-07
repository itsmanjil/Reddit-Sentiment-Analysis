from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/sentiment/', include('app.urls')),
    path('api/', include('app_api.urls')),
    path('api/user/', include('users.urls')),
    
]
