# Kaggle Instacart Classification

I built models to classify whether or not items in a user's order history will be in their most recent order, basically recreating the [Kaggle Instacart Market Basket Analysis Competition](https://www.kaggle.com/c/instacart-market-basket-analysis/overview/description). Because the full dataset was too large to work with on my older Macbook, I loaded the data into a SQL database on an AWS EC2 instance. This setup allowed me to easily query subsets of the data in order to do all of my preliminary development. I then cleaned up my work and wrote it into [a Python script called 'build_models.py'](code/build_models.py) that can be easily run through a notebook or the command line. Once I was ready to scale up to the full dataset, I simply ran the build_models script on a 2XL EC2 instance and brought the resulting models back into [my 'kaggle_instacart' notebook](kaggle_instacart.ipynb) for test set evaluation. This github repo also contains my more unstructured working files in [the dev directory](dev).

### Feature Engineering

I spent the majority of my time on this project engineering features from the basic dataset. After creating several features, I tested different combinations of them on a small subset of the data in order to eliminate any that seemed to have no effect on model output. After paring down features I ended up training and testing my final models on the following columns:
* **percent_in_user_orders**: Percent of a user's orders in which an item appears
* **percent_in_all_orders**: Percent of all orders in which an item appears
* **in_last_cart**: 1 if an item appears in a user's most recent prior order, 0 if not
* **in_last_five**: Number of orders in a user's five most recent prior orders in which an item appears
* **total_user_orders**: Total number of orders placed by a user
* **mean_orders_between**: Average number of orders between which an item appears in a user's order
* **mean_days_between**: Average number of days between which an item appears in a user's order
* **orders_since_newest**: Number of orders between the last user order containing an item and the most recent order
* **days_since_newest**: Number of days between the last user order containing an item and the most recent order
* **product_reorder_proba**: Probability that any user reorders an item
* **user_reorder_proba**: Probability that a user reorders any item
* **mean_cart_size**: Average user cart (aka order) size
* **mean_cart_percentile**: Average percentile of user cart add order for an item
* **mean_hour_of_week**: Average hour of the week that a user orders an item (168 hours in a week)
* **newest_cart_size**: Number of items in the most recent cart
* **newest_hour_of_week**: Hour of the week that the most recent order was placed
* **cart_size_difference**: Absolute value of the difference between the average size of the orders containing an item and the size of the most recent order
* **hour_of_week_difference**: Absolute value of the difference between the average hour of the week in which a user purchases an item and the hour of the week of the most recent order
