# Introduction
This repository is a small but useful logging tool for developing. It is originated from the Yash Prakash at towardsdatascience.com.

I put it here for easy use. 

# Usage
In the `main` function, you can assign the name of the log file.
```python
import logger
log = logger.setup_app_level_logger(file_name="app_debug.log")
```

In the `modules`, you just need to include the following two lines in the front of the file:
```python
import logger
log = logger.get_logger(__name__)
```

then, you can use the `log` as usually. The output of `log` will redirect to the file, which is much easier to debug.



# Reference
https://towardsdatascience.com/the-reusable-python-logging-template-for-all-your-data-science-apps-551697c8540

https://github.com/yashprakash13/Python-Cool-Concepts/tree/main/logging_template
