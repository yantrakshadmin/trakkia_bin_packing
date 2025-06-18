from django.urls import path
from .views import PackingAPIView
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('pack-items/', PackingAPIView.as_view(), name='pack-items'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)