import tensorflow as tf


class AETrain:
    @staticmethod
    def compute_loss(model, x):
        loss_object = tf.keras.losses.BinaryCrossentropy()
        z = model.encode(x)
        x_logits = model.decode(z)
        loss = loss_object(x, x_logits)
        return loss

    @staticmethod
    def compute_gradients(model, x):
        with tf.GradientTape() as tape:
            loss = AETrain.compute_loss(model, x)
        return tape.gradient(loss, model.trainable_variables), loss

    @staticmethod
    def apply_gradients(optimizer, gradients, variables):
        optimizer.apply_gradients(zip(gradients, variables))


class VAETrain:
    @staticmethod
    def compute_loss(model, x):
        mean, logvar = model.encode(x)
        z = model.reparameterize(mean, logvar)
        x_logits = model.decode(z)

        # cross_ent = - marginal likelihood
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logits, labels=x)
        marginal_likelihood = - tf.reduce_sum(cross_ent, axis=1)
        marginal_likelihood = tf.reduce_mean(marginal_likelihood)

        KL_divergence = tf.reduce_sum(mean ** 2 + tf.exp(logvar) - logvar - 1, axis=1)
        KL_divergence = tf.reduce_mean(KL_divergence)

        ELBO = marginal_likelihood - KL_divergence
        loss = -ELBO
        return loss

    @staticmethod
    def compute_gradients(model, x):
        with tf.GradientTape() as tape:
            loss = VAETrain.compute_loss(model, x)
        return tape.gradient(loss, model.trainable_variables), loss

    @staticmethod
    def apply_gradients(optimizer, gradients, variables):
        optimizer.apply_gradients(zip(gradients, variables))


class CVAETrain:
    @staticmethod
    def compute_loss(model, x, y):
        mean, logvar = model.encode(x, y)
        z = model.reparameterize(mean, logvar)
        x_logits = model.decode(z, y)

        # cross_ent = - marginal likelihood
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logits, labels=x)
        marginal_likelihood = - tf.reduce_sum(cross_ent, axis=1)
        marginal_likelihood = tf.reduce_mean(marginal_likelihood)

        KL_divergence = tf.reduce_sum(mean ** 2 + tf.exp(logvar) - logvar - 1, axis=1)
        KL_divergence = tf.reduce_mean(KL_divergence)

        ELBO = marginal_likelihood - KL_divergence
        loss = -ELBO
        return loss

    @staticmethod
    def compute_gradients(model, x, y):
        with tf.GradientTape() as tape:
            loss = CVAETrain.compute_loss(model, x, y)
        return tape.gradient(loss, model.trainable_variables), loss

    @staticmethod
    def apply_gradients(optimizer, gradients, variables):
        optimizer.apply_gradients(zip(gradients, variables))
