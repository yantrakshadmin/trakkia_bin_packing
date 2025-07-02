from django.urls import path
from .views import PackingAPIView, home, serve_plot
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import TemplateView


urlpatterns = [
    path('', home, name='home'),
    path('pack-items/', PackingAPIView.as_view(), name='pack-items'),
    path("plot/<str:plot_id>/", serve_plot, name="serve_plot"),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)