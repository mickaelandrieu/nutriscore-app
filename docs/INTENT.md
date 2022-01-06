## Evaluating the Nutriscore : encouraging the contribution on OpenFoodFacts

According to the scientific documentation of Nutriscore, it is calculated from the nutritional declaration per 100g of a product sold. It is a score made up of points in favor of health from which points are subtracted against health.

According to an article on the [Santé Publique France](https://www.santepubliquefrance.fr/en/nutri-score) site, it is considered that the amount of the following items should be limited:

* calories (energy)
* saturated fatty acids
* sugars
* salt

Conversely, the model encourages / rewards the following elements:

* fibers
* protein
* fruit
* vegetables, legumes
* nuts
* rapeseed and olive oils

Here, we are talking about essentially processed products but we consider, for example, that most cheeses are also processed products.

According to the Excel file accessible on their site, it seems that there are special categories of calculation: cheese, added fats and drinks.

> In this project, we are going to build a model only valid in the general case!

The project will consist in verifying whether it is possible to build a sufficiently reliable statistical model from the data at our disposal.

We have cleaned and prepared data from the [OpenFoodFacts website](https://world.openfoodfacts.org/data).
For instance, we have kept the nutritional data for 100g, the nutriscore (in letter and score format) and the product category because we must consider it to discriminate special cases (drinks for example).

The difficulty with our project is that the product category is rarely defined and that we do not always have access to all the required nutritional data in order to build the nutriscore : this is why a machine learning approach is used here.
