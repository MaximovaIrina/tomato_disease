# PlantDisease
Методы машинного обучения в задаче обнаружения и классификации болезней листьев томатов

## Описание базы данных
В проекте используется [Plant Village Dataset], содержащий 6000 отсегментированных изображений больных и здоровых листьев томатов, сбалансированно
разбитый на 6 категорий:
<p align="center">
<img src="reports/readme_design/group_names.jpg" alt="Круговая диаграмма распределения изображений по группам" height="180"/>
<img src="reports/readme_design/example_images.png" alt="Примеры листьев томатов, пораженных исследуемыми заболеваниями" height="150"/>
</p>

Характеристики изображений:
* Разрешение: 256х256
* Глубина цвета: 8bit
* Формат хранения: .jpeg
* Цветовой профиль: RGB

## Предобработка данных
**1. Удаление черного фона.** Наличие большой черной области сильно искажает сатистические и  тектурные признаки изображения. Обнаружено, что на края листьев и прелегающей области находится много околонулевых писелей - вероятно это эффект предварительной сегментации. Поэтому фоновыми считаются пиксели с интесивностью, меньше 10. Данный порог подобран эмпирически. <p align="center"><img src="reports/readme_design/remove_10_pixel.png" alt="Фоновые пиксели" height="200"/></p>
**2. Фильтрация выборки здоровых изображений.** Обнаружено, что выборка здоровых листьев состоит из двух 2 типов изображений – гладкие, равномерно освещенные листья и листья с фактурой и тенью. Для дальнейшего исследования решено оставить те изображения, что ближе к реальности – с фактурой и тенью. Отсеивая все изображения, чьи стандартные отклонения
меньше 30,  будет обрезано распределение по данному признаку. Чтобы этого не допустить добавлена проверка на однородность: HOM > 0.4. Все пороги отсеивания подобраны из вида распределений соответствующего признака.  
<p align="center">
<img src="reports/readme_design/two_types_healthy.png" alt="Два типа изображений здоровых листьев" height="150"/>  
</p>  
<p align="center">
<img src="reports/readme_design/stat_healthy.png" alt="Статистика интесивностей пикселей здоровой выборки" height="150"/>
<img src="reports/readme_design/healthy_hom.png" alt="Распределение оценки однородности изображений для здоровой выборки" height="140"/>
</p>

## Извлечение признаков  
Для обучения и тестирования классификаторов из изображений извлекались следующие группы признаков: 
- **STAT**. Статистические характеристики интесивности пикселей изображения:  
<p align="center">
  <img src="https://latex.codecogs.com/gif.latex?\mu&space;=&space;\frac{1}{N}&space;\sum_{i}^{}L_i" title="\mu = \frac{1}{N} \sum_{i}^{}L_i" />
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://latex.codecogs.com/gif.latex?\sigma&space;=&space;\sqrt&space;{\frac{1}{N}\sum&space;(L_i&space;-&space;\mu)^2}" title="\sigma = \sqrt {\frac{1}{N}\sum (L_i - \mu)^2}" />
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://latex.codecogs.com/gif.latex?min_{normalized}&space;=&space;\frac{\mu&space;-&space;min\left\{{L_i}\right\}}{\sigma}" title="min_{normalized} = \frac{\mu -      min\left\{{L_i}\right\}}{\sigma}"/>
   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://latex.codecogs.com/gif.latex?max_{normalized}&space;=&space;\frac{max\left\{{L_i}\right\}&space;-&space;\mu}{\sigma}" title="max_{normalized} =                  \frac{max\left\{{L_i}\right\} - \mu}{\sigma}" />
</p>
- **HIST**. Бины квантованной гостограммы изображений:   
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?Q&space;=&space;\begin{cases}&space;[\mu&space;-&space;2\sigma,&space;\mu&space;-&space;\sigma),&space;&&space;\mbox{0th&space;bin&space;}&space;\\&space;[\mu&space;-&space;\sigma,&space;\mu)&space;&&space;\mbox{1st&space;bin&space;}&space;\\&space;[\mu,&space;\mu&space;&plus;&space;\sigma)&space;&&space;\mbox{2nd&space;bin&space;}&space;\\&space;[\mu&space;&plus;&space;\sigma,&space;\mu&space;&plus;&space;2\sigma)&space;&&space;\mbox{3rd&space;bin&space;}&space;\\&space;\end{cases}" title="Q = \begin{cases} [\mu - 2\sigma, \mu - \sigma), & \mbox{0th bin } \\ [\mu - \sigma, \mu) & \mbox{1st bin } \\ [\mu, \mu + \sigma) & \mbox{2nd bin } \\ [\mu + \sigma, \mu + 2\sigma) & \mbox{3rd bin } \\ \end{cases}" />
</p>
- **GLCM**. Текстурные признаки изображений на базе матрицы [GLCM]:  


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

[GLCM]: https://prism.ucalgary.ca/bitstream/handle/1880/51900/texture%20tutorial%20v%203_0%20180206.pdf?sequence=11&isAllowed=y
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
