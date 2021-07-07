# PlantDisease
Курсовая работа на тему "Методы машинного обучения в задаче обнаружения и
классификации болезней листьев томатов"

## Описание базы данных
В проекте используется [Plant Village Dataset], содержащий 6000 отсегментированных изображений больных и здоровых листьев томатов, сбалансированно
разбитый на 6 категорий:
<p align="center">
<img src="materials/readme_design/group_names.jpg" alt="Круговая диаграмма распределения изображений по группам" width="400" class="center"/>
</p>

Характеристики изображений:
* Разрешение: 256х256
* Глубина цвета: 8bit
* Формат хранения: .jpeg
* Цветовой профиль: RGB

## Описание модулей
* model.py - содержит базовые модели: экстрактор функций и полную модель HealthyPlant
* dataset.py - операции с датасетом  
* evaluate.py - скрипт для получения метрик от всех классификаторов
* features.py - извлечение и сохранение признаков  
* inference.py - скрипт для получения предсказания на конкретном изображении полной модели
* single_clf_cross_val.py - кросс-валидация на частном примере
* utils.py - разные функции, используемые в других скриптах

## Результаты
Результаты подробно описаны в [отчете за 2й семестр](materials/reports/report_2_sem.pdf 'отчет'). Реалицация в вектке master.   
Удаление лишних признаков см в [отчете за 3й семестр](materials/reports/report_3_sem.pdf 'отчет'). Реалицация в вектке pca.  
  
## Tools & Docs & Article:
1. [GLCM Texture: A Tutorial]
1. [PyTorch]
1. [Scikit Learn algorithms]
1. [Scikit-image. Greycomatrix]
1. [Scikit-image. Greycoprops]
1. [Plant Village Dataset]
1. [RGB+NIR Extraction From A Single RAW Image]
1. [Measuring Vegetation (NDVI & EVI)]
1. [PCA in detail]
1. [PCA understandingly]

## Планы на будущее
1. Попробовать эквализировать изображения. Посмотреть как изменится кач-во. 
1. Добавить синий и/или зеленый канал в рассмотрение
1. Возможно квантовать на больше е число уровней
1. В отчете : cравнить изображения фич (18 стр 3 сем) с оставленными в конце GLCM фичами. Какой визуальный вклад?
1. Архитектура признаков. Использовать не все корреляции по отдельности а их сумму. 
1. Что на счет статичтических GlCM признаков
1. Добавить confusion matrix
1. Провести анализ отдельно над каждым типом признаков. Построить гистограмммы для статистических признаков. Исследовать оптимальное число уровней квантования для hist и stat.
1. Возможно оставить лишь одну задачу - классификации, когда переходим к локальным признакам. А в момент определения лучшего классификатора, уменьшать размерность пространства признаков только относительно него.

[GLCM Texture: A Tutorial]: https://prism.ucalgary.ca/bitstream/handle/1880/51900/texture%20tutorial%20v%203_0%20180206.pdf?sequence=11&isAllowed=y
[PyTorch]: https://pytorch.org/docs/stable/index.html
[Scikit Learn algorithms]: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree
[Scikit-image. Greycomatrix]:https://scikit-image.org/docs/dev/api/skimage.feature.html?highlight=skimage%20feature#skimage.feature.greycomatrix
[Scikit-image. Greycoprops]:https://scikit-image.org/docs/dev/api/skimage.feature.html?highlight=skimage%20feature#skimage.feature.greycoprops
[Plant Village Dataset]:https://github.com/spMohanty/PlantVillage-Dataset
[Bacterial spot]:https://www2.ipm.ucanr.edu/agriculture/tomato/bacterial-spot/
[Early blight]:https://www2.ipm.ucanr.edu/agriculture/tomato/Early-Blight/
[Late blight]:https://www2.ipm.ucanr.edu/agriculture/tomato/Late-Blight/
[Yellow Leaf Curl Virus]:https://www2.ipm.ucanr.edu/agriculture/tomato/tomato-yellow-leaf-curl/
[Septoria leaf spot]:http://ipm.ucanr.edu/PMG/GARDEN/PLANTS/DISEASES/septorialfspot.html
[RGB+NIR Extraction From A Single RAW Image]: http://aggregate.org/dit/rgbnir/
[Measuring Vegetation (NDVI & EVI)]:https://www.earthobservatory.nasa.gov/features/MeasuringVegetation
[PCA in detail]:https://e-learning.unn.ru/pluginfile.php/41225/mod_resource/content/4/%2B%2B%20%20%20%20%D0%9B%D0%B5%D0%BA%D1%86%D0%B8%D1%8F%202%20%D0%9C%D0%93%D0%9A.pdf
[PCA understandingly]:https://habr.com/ru/post/304214/
