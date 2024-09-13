from django.urls import path
from .views import pack_items_view
from django.conf import settings
from django.conf.urls.static import static


urlpatterns = [
    path('', pack_items_view, name='packing_form'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)