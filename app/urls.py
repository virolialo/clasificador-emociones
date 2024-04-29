from django.urls import path
from . import views

urlpatterns = [
    path('models', views.botones_modelos, name='botones_modelos'),
    path('resultNB/', views.boton_viewNB, name='boton_viewNB'),
    path('resultGNB/', views.boton_viewGNB, name='boton_viewGNB'),
    path('resultDT/', views.boton_viewDT, name='boton_viewDT'),
    path('resultknn/', views.boton_viewKnn, name='boton_viewKnn'),
    path('', views.pagina_principal, name='pagina_inicial'),

]