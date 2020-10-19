class EarlyStopping:
    """A simple callback to stop NetKet if there are no improvements in the training"""

    def __init__(self, min_delta=0, patience=0, baseline=None):
        """
        Constructs a new EarlyStopping object that monitors whether a driver is improving
        over optimisation epochs based on `driver._loss_name`.

        Args:
            min_delta: Minimum change in the monitored quantity to qualify as an improvement.
            patience: Number of epochs with no improvement after which training will be stopped.
            baseline: Baseline value for the monitored quantity. Training will stop if the driver
                hits the baseline.
        """
        self.__min_delta = min_delta
        self.__patience = patience
        self.__baseline = baseline
        self.__best_val = None
        self.__best_iter = None

    def earlystopping(self, step, log_data, driver):
        """
        A boolean function that determines whether or not to stop training.
        
        Args:
            step: An integer corresponding to the step (iteration or epoch) in training.
            log_data: A dictionary containing log data for training.
            driver: A NetKet variational driver.

        Returns:
            A boolean. If True, training continues, else, it does not.
        """
        loss = log_data[driver._loss_name].mean.real
        if step == 0:
            self.__best_val = loss
            self.__best_iter = step
        else:
            if loss <= self.__best_val:
                self.__best_val = loss
                self.__best_iter = step
        if self.__baseline is not None:
            if loss <= self.__baseline:
                return False
        if (
            step - self.__best_iter > self.__patience
            and loss > self.__best_val - self.__min_delta
        ):
            return False
        else:
            return True
