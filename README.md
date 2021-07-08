# PlantDisease
Методы машинного обучения в задаче обнаружения и классификации болезней листьев томатов

## Описание базы данных
В проекте используется [Plant Village Dataset], содержащий 6000 отсегментированных изображений больных и здоровых листьев томатов, сбалансированно
разбитый на 6 категорий:
<p align="center">
<img src="reports/readme_design/group_names.jpg" alt="Круговая диаграмма распределения изображений по группам" height="170"/>
<img src="reports/readme_design/example_images.png" alt="Примеры листьев томатов, пораженных исследуемыми заболеваниями" height="150"/>
</p>

Характеристики изображений:
* Разрешение: 256х256
* Глубина цвета: 8bit
* Формат хранения: .jpeg
* Цветовой профиль: RGB

## Предобработка данных
**1. Удаление черного фона.** Наличие большой черной области сильно искажает сатистические и  тектурные признаки изображения. Обнаружено, что на края листьев и прелегающей области находится много околонулевых писелей - вероятно это эффект предварительной сегментации. Поэтому фоновыми считаются пиксели с интесивностью, меньше 10. Данный порог подобран эмпирически. <p align="center"><img src="reports/readme_design/remove_10_pixel.png" alt="Фоновые пиксели" height="200"/></p>
**2. Фильтрация выборки здоровых изображений.** Обнаружено, что выборка здоровых листьев состоит из двух типов изображений – гладкие, равномерно освещенные листья и листья с фактурой и тенью. Для дальнейшего исследования решено оставить те изображения, что ближе к реальности – с фактурой и тенью. Отсеивая все изображения, чьи стандартные отклонения
меньше 30,  будет обрезано распределение по данному признаку. Чтобы этого не допустить добавлена проверка на однородность: HOM > 0.4. Все пороги отсеивания подобраны из вида распределений соответствующего признака.  
<p align="center">
<img src="reports/readme_design/two_types_healthy.png" alt="Два типа изображений здоровых листьев" height="150"/>  
</p>  
<p align="center">
<img src="reports/readme_design/stat_healthy.png" alt="Статистика интесивностей пикселей здоровой выборки" height="150"/>
<img src="reports/readme_design/healthy_hom.png" alt="Распределение оценки однородности изображений для здоровой выборки" height="140"/>
</p>

## Выбор источников признаков
В работе рассатиривает два "источника" признаков:  
**1. RED канал RGB изображения.** Пигмент листьев растений, хлорофилл, сильно поглощает видимый свет (от 0,4 до 0,7 мкм) для использования в фотосинтезе. При заболевании образование хлорофилла в листьях нарушается, что приводит к увеличению отражения данных длин волн. Это выражается более явно в красном канале. Поэтому возникает предположение, что признаков, извлеченных из красного канала изображения, окажется достаточно, чтобы с успехом диагностировать заболевание.  <p align="center"><img src="reports/readme_design/R_G_B_example_img.png" alt="Разложене по каналам" height="150"/></p>
**2. NDVI образы изображений.** [NDVI] – нормализованный вегетационный индекс, по которому можно судить о количестве и качестве растительности. NDVI рассчитыается в красном и инфракрасном диапазоне. Поскольку мы работаем с RGB изображениями, то в чистом виде информации об инфракрасном диапазоне нет. Вместо этого мы обладаем информацией о зеленой области спектра (0.5-0.6 мкм). Так как растения отражают зеленый свет сильнее красного предлагается ввести аналог вегетационного индекса на основе зеленого и красного каналов:
<p align="center">
<img src="https://latex.codecogs.com/gif.latex?NDVI_G&space;=&space;\frac{GREEN&space;-&space;RED&space;}{GREEN&space;&plus;&space;RED}" title="NDVI_G = \frac{GREEN - RED }{GREEN + RED}" />
</p>
<p align="center">
  <img src="reports/readme_design/plant_mirror.png" alt="Отражение" height="200"/>
  <img src="reports/readme_design/NDVI_G.png" alt="Примеры NDVI_G образов изображений" height="200"/>
</p>

## Извлечение признаков  
Для обучения и тестирования классификаторов из изображений извлекались следующие группы признаков: 
- **STAT**. Статистические характеристики интесивности пикселей изображения: <p align="center"><img src="https://latex.codecogs.com/gif.latex?\mu&space;=&space;\frac{1}{N}&space;\sum_{i}^{}L_i" title="\mu = \frac{1}{N} \sum_{i}^{}L_i" />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://latex.codecogs.com/gif.latex?\sigma&space;=&space;\sqrt&space;{\frac{1}{N}\sum&space;(L_i&space;-&space;\mu)^2}" title="\sigma = \sqrt {\frac{1}{N}\sum (L_i - \mu)^2}" /><br><img src="https://latex.codecogs.com/gif.latex?min_{normalized}&space;=&space;\frac{\mu&space;-&space;min\left\{{L_i}\right\}}{\sigma}" title="min_{normalized} = \frac{\mu -   min\left\{{L_i}\right\}}{\sigma}"/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://latex.codecogs.com/gif.latex?max_{normalized}&space;=&space;\frac{max\left\{{L_i}\right\}&space;-&space;\mu}{\sigma}" title="max_{normalized} = \frac{max\left\{{L_i}\right\} - \mu}{\sigma}" /></p>
- **HIST**. Бины квантованной гостограммы изображений: <p align="center"><img src="https://latex.codecogs.com/gif.latex?Q&space;=&space;\begin{cases}&space;[\mu&space;-&space;2\sigma,&space;\mu&space;-&space;\sigma),&space;&&space;\mbox{0th&space;bin&space;}&space;\\&space;[\mu&space;-&space;\sigma,&space;\mu)&space;&&space;\mbox{1st&space;bin&space;}&space;\\&space;[\mu,&space;\mu&space;&plus;&space;\sigma)&space;&&space;\mbox{2nd&space;bin&space;}&space;\\&space;[\mu&space;&plus;&space;\sigma,&space;\mu&space;&plus;&space;2\sigma)&space;&&space;\mbox{3rd&space;bin&space;}&space;\\&space;\end{cases}" title="Q = \begin{cases} [\mu - 2\sigma, \mu - \sigma), & \mbox{0th bin } \\ [\mu - \sigma, \mu) & \mbox{1st bin } \\ [\mu, \mu + \sigma) & \mbox{2nd bin } \\ [\mu + \sigma, \mu + 2\sigma) & \mbox{3rd bin } \\ \end{cases}" /> &nbsp;&nbsp; <img src="reports/readme_design/hist_fe_groups.png" alt="Гистограммы интенсивностей пикселей для каждой группа болезней" height="100"/></p>
- **GLCM**. Текстурные признаки изображений на базе матрицы [GLCM]: <p align="center"><img src="https://latex.codecogs.com/gif.latex?energy&space;=&space;\sum_{i,j}^{}(P_{ij})^2" title="energy = \sum_{i,j}^{}(P_{ij})^2" />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://latex.codecogs.com/gif.latex?entropy&space;=&space;\sum_{i,j}^{}-ln(P_{ij})P_{ij}" title="entropy = \sum_{i,j}^{}-ln(P_{ij})P_{ij}" />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<img src="https://latex.codecogs.com/gif.latex?contrast&space;=&space;\sum_{i,j}^{}&space;P_{ij}(i-j)^2" title="contrast = \sum_{i,j}^{} P_{ij}(i-j)^2" /><br><img src="https://latex.codecogs.com/gif.latex?homogeneity&space;=&space;\sum_{i,j}^{}&space;\frac{P_{ij}}{1&plus;(i-j)^2}" title="homogeneity = \sum_{i,j}^{} \frac{P_{ij}}{1+(i-j)^2}" />&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://latex.codecogs.com/gif.latex?correlation&space;=&space;\sum_{i,j}^{}&space;P_{ij}&space;\frac{(i-\mu)(j-\mu)}{\sigma^2}" title="correlation = \sum_{i,j}^{} P_{ij} \frac{(i-\mu)(j-\mu)}{\sigma^2}" /></p> Для вычисления GLCM необходимо задать два параметра: *d* - расстояние межде пикселями и *φ* - направление от первого пикселя ко второму. Для выбора оптимальных параметров *d* и *φ*, способных наилучшим образом отделять классы, использовалась информация о косинусном расстоянии между средними векторами глобальных текстурных признаков. Чем меньше угол, тем выше сходство. Для каждого класса вычислен средний вектор глобальных текстурных признаков и найдено среднее значение косинуса с векторами остальных классов при заданной паре параметров (d, φ). <p align="center"> <img src="reports/readme_design/cosine.png" alt="Косинус" height="200"/></p>В качестве лучшего расстояния можно выделить d=4. Признаки, рассчитанные при таком значении d, имеют наименьшее косинусное сходство, значит более разнообразно описывают классы. Относительно параметра φ сложно выделить лучшее значение

**Все признаки извлекались без учета фоновых пикселей**

## Способы извлечения признаков
Поскольку признаки заболевания растения проявляются в разных областях листа, то для того, чтобы обеспечить устойчивость показаний признаков к пространственному сдвигу,
помимо глобальных признаков, будут рассматриваться и локальные признаки.  
- **global** - признаки, извлеченные неким оператором над всем изображением сразу. 
- **local** - признаки, извлеченные под маской инструмента размером 17х17 пикселей.  
Алгоритм извлечения локальных призаков: 
<p align="center">
  <img src="reports/readme_design/algorithm_for_extract_local_feach.png" alt="Локальные признаки здорового листа" height="300"/>
</p>

Визуализация локальных признаков: 
<p align="center">
  <img src="reports/readme_design/local_feach_healthy.png" alt="Локальные признаки здорового листа" height="250"/>
  <img src="reports/readme_design/local_feach_disease.png" alt="Локальные признаки больного листа" height="250"/>
</p>

## Классификаторы и механизм их обучения

## Результаты детектирования и классификации


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
[NDVI]:https://earthobservatory.nasa.gov/features/MeasuringVegetation
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
