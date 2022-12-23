import os
from utils_imagnet.logging_imagenet import DEBUG, do_logging_system_use


def do_training_loop(training_data, model, loss_fn, optimizer, device):
    data_inputs, data_labels = training_data

    # Move input data to device (only strictly necessary if we use GPU)
    data_inputs = data_inputs.to(device)
    data_labels = data_labels.to(device)

    # Run the model on the input data
    training_outputs = model(data_inputs)
    # Calculate the loss
    loss = loss_fn(training_outputs, data_labels)

    # Perform backpropagation
    # Before calculating the gradients, we need to ensure that they are all zero.
    # The gradients would not be overwritten, but actually added to the existing ones.
    optimizer.zero_grad()
    # Perform backpropagation
    loss.backward()

    #  Update the parameters
    optimizer.step()

    # Calculate the total and average loss
    return loss


def training_loop(
    train_loader, current_epoch, model, optimizer, loss_fn, device, data_dir, logger
):

    total_train_loss = 0.0
    for cur_train_iter, training_data in enumerate(train_loader):
        print(
            f"In iteration: {cur_train_iter} and epoch: {current_epoch} of training loader"
        )

        if DEBUG:
            logger.debug("##START OF TRAINING ITERATION. \n\n\n ")
            logger.debug(
                f"Entering iteration: {cur_train_iter} and epoch: {current_epoch} of training loader\n"
            )
            do_logging_system_use(
                logger=logger,
                current_epoch=current_epoch,
                current_dir=os.getcwd(),
                data_dir=data_dir,
            )

        total_train_loss += do_training_loop(
            training_data=training_data,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        if DEBUG:
            logger.debug(
                f"Exiting iteration: {cur_train_iter} and epoch: {current_epoch} of training loader with loss: {total_train_loss}\n"
            )
            do_logging_system_use(
                logger=logger,
                current_epoch=current_epoch,
                current_dir=os.getcwd(),
                data_dir=data_dir,
            )
            logger.debug("##STOP OF TRAINING ITERATION. \n\n\n ")

    return total_train_loss


def validation_loop(
    validation_loader,
    current_epoch,
    model,
    loss_fn,
    device,
    data_dir,
    logger,
):
    total_validation_loss = 0.0
    for cur_validation_iter, validation_data in enumerate(validation_loader):
        print(
            f"In iteration: {cur_validation_iter} and epoch: {current_epoch} of validation loader"
        )
        if DEBUG:
            assert logger is not None
            logger.debug("##START OF VALIDATION ITERATION. \n\n\n ")
            logger.debug(
                f"Entering iteration: {cur_validation_iter} and epoch: {current_epoch} of validation loader\n"
            )
            do_logging_system_use(
                logger=logger,
                current_epoch=current_epoch,
                current_dir=os.getcwd(),
                data_dir=data_dir,
            )

        total_validation_loss += do_validation_loop(
            validation_data=validation_data, model=model, loss_fn=loss_fn, device=device
        )

        if DEBUG:
            assert logger is not None
            logger.debug(
                f"Exiting iteration: {cur_validation_iter} and epoch: {current_epoch} of validation loader\n"
            )
            do_logging_system_use(
                logger=logger,
                current_epoch=current_epoch,
                current_dir=os.getcwd(),
                data_dir=data_dir,
            )
            logger.debug("##STOP OF VALIDATION ITERATION. \n\n\n ")

    return total_validation_loss


def do_validation_loop(validation_data, model, loss_fn, device):
    validation_inputs, validation_labels = validation_data
    validation_inputs = validation_inputs.to(device)
    validation_labels = validation_labels.to(device)
    validation_outputs = model(validation_inputs)
    validation_loss = loss_fn(validation_outputs, validation_labels)
    return validation_loss


def do_early_stopping(
    early_stopper, validation_loss, model, logger, current_epoch, data_dir
):
    # early_stopping needs the validation loss to check if it has
    # decresed, and if it has, it will make a checkpoint of the current
    # model. If the best validation value hasn't improved in some time
    # it sets the early_stop instance variable, so we know that we need
    # to stop.
    early_stopper(validation_loss, model)
    if early_stopper.early_stop:
        if DEBUG:
            assert logger is not None
            logger.debug(f"Early stopping in epoch {current_epoch}")
            do_logging_system_use(
                logger=logger,
                current_epoch=current_epoch,
                current_dir=os.getcwd(),
                data_dir=data_dir,
            )

    return early_stopper.early_stop
