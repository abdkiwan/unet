
import tensorflow as tf
import layers as layers

class UNet(object):

    def __init__(self, img_size):
        self.img_size = img_size
    
    def build(self, X):
        """
        Build the graph of network:
        ----------
        Args:
            X: Tensor, [1, height, width, 3]
        Returns:
            logits: Tensor, predicted annotated image flattened 
                                  [1 * height * width,  num_classes]
        """

        dropout_prob = tf.where(True, 0.2, 1.0)

        # Left Side
        down_1_conv_1 = layers.Conv2d(X,  [3, 3], 64, 'down_1_conv_1')
        down_1_conv_2 = layers.Conv2d(down_1_conv_1, [3, 3], 64, 'down_1_conv_2')
        down_1_pool   = layers.Maxpool(down_1_conv_2, [2, 2], 'down_1_pool')

        down_2_conv_1 = layers.Conv2d(down_1_pool, [3, 3], 128, 'down_2_conv_1')
        down_2_conv_2 = layers.Conv2d(down_2_conv_1, [3, 3], 128, 'down_2_conv_2')
        down_2_pool   = layers.Maxpool(down_2_conv_2, [2, 2], 'down_2_pool')

        down_3_conv_1 = layers.Conv2d(down_2_pool, [3, 3], 256, 'down_3_conv_1')
        down_3_conv_2 = layers.Conv2d(down_3_conv_1, [3, 3], 256, 'down_3_conv_2')
        down_3_pool   = layers.Maxpool(down_3_conv_2, [2, 2], 'down_3_pool')
        down_3_drop   = layers.Dropout(down_3_pool, dropout_prob, 'down_3_drop')

        down_4_conv_1 = layers.Conv2d(down_3_drop, [3, 3], 512, 'down_4_conv_1')
        down_4_conv_2 = layers.Conv2d(down_4_conv_1, [3, 3], 512, 'down_4_conv_2')
        down_4_pool   = layers.Maxpool(down_4_conv_2, [2, 2], 'down_4_pool')
        down_4_drop   = layers.Dropout(down_4_pool, dropout_prob, 'down_4_drop')

        down_5_conv_1 = layers.Conv2d(down_4_drop, [3, 3], 1024, 'down_5_conv_1')
        down_5_conv_2 = layers.Conv2d(down_5_conv_1, [3, 3], 1024, 'down_5_conv_2')
        down_5_drop   = layers.Dropout(down_5_conv_2, dropout_prob, 'down_5_drop')

        # Right Side
        up_6_deconv = layers.Deconv2d(down_5_drop, 2, 'up_6_deconv')
        up_6_concat = layers.Concat(up_6_deconv, down_4_conv_2, 'up_6_concat')
        up_6_conv_1 = layers.Conv2d(up_6_concat, [3, 3], 512, 'up_6_conv_1')
        up_6_conv_2 = layers.Conv2d(up_6_conv_1, [3, 3], 512, 'up_6_conv_2')
        up_6_drop   = layers.Dropout(up_6_conv_2, dropout_prob, 'up_6_drop')

        up_7_deconv = layers.Deconv2d(up_6_drop, 2, 'up_7_deconv')
        up_7_concat = layers.Concat(up_7_deconv, down_3_conv_2, 'up_7_concat')
        up_7_conv_1 = layers.Conv2d(up_7_concat, [3, 3], 256, 'up_7_conv_1')
        up_7_conv_2 = layers.Conv2d(up_7_conv_1, [3, 3], 256, 'up_7_conv_2')
        up_7_drop   = layers.Dropout(up_7_conv_2, dropout_prob, 'up_7_drop')

        up_8_deconv = layers.Deconv2d(up_7_drop, 2, 'up_8_deconv')
        up_8_concat = layers.Concat(up_8_deconv, down_2_conv_2, 'up_8_concat')
        up_8_conv_1 = layers.Conv2d(up_8_concat, [3, 3], 128, 'up_8_conv_1')
        up_8_conv_2 = layers.Conv2d(up_8_conv_1, [3, 3], 128, 'up_8_conv_2')

        up_9_deconv = layers.Deconv2d(up_8_conv_2, 2, 'up_9_deconv')
        up_9_concat = layers.Concat(up_9_deconv, down_1_conv_2, 'up_9_concat')
        up_9_conv_1 = layers.Conv2d(up_9_concat, [3, 3], 64, 'up_9_conv_1')
        up_9_conv_2 = layers.Conv2d(up_9_conv_1, [3, 3], 64, 'up_9_conv_2')

        score  = layers.Conv2d(up_9_conv_2, [1, 1], 1, 'score')
        logits = tf.reshape(score, (-1, 1))

        return logits

    def train(self, X_train, y_train, X_test, y_test, num_epochs, learning_rate, model_save_dir):

        sess = tf.Session()
        X = tf.placeholder('float32', [1, self.img_size, self.img_size, 3], name='input_image')        
        y = tf.placeholder('float32', [self.img_size*self.img_size, 1], name='target_segment')
        
        # Forward        
        logits = self.build(X)

        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y))

        # Accuracy
        predictions = tf.round(tf.nn.sigmoid(logits))
        correct_predictions = tf.equal(predictions, y)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

        # Optimizer
        global_step = tf.Variable(0, name = 'global_step', trainable = False)
        optimizer   = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op    = optimizer.minimize(loss, global_step = global_step)

        # To save the weights later
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())

        # To store the loss values for plotting later
        train_loss_values = []
        test_loss_values = []

        for epoch in range(num_epochs):
            
            step = tf.train.global_step(sess, global_step)

            epoch_train_loss = 0
            
            for epoch_x, epoch_y in zip(X_train, y_train):
                epoch_x = epoch_x.reshape((1, self.img_size, self.img_size, 3))
                epoch_y = epoch_y.reshape((self.img_size*self.img_size, 1))

                # Returning the loss value on train samples
                _, l = sess.run([train_op, loss], feed_dict={X: epoch_x, y: epoch_y})
                epoch_train_loss += l
            
            # Calculate the loss value of the epoch
            epoch_train_loss = epoch_train_loss/len(X_train)
            train_loss_values.append(epoch_train_loss)
        
            # Calculate loss on test dataset
            epoch_test_loss = 0
            for epoch_x, epoch_y in zip(X_test, y_test):
                epoch_x = epoch_x.reshape((1, self.img_size, self.img_size, 3))
                epoch_y = epoch_y.reshape((self.img_size*self.img_size, 1))
                
                epoch_test_loss += sess.run(loss, feed_dict={X: epoch_x, y: epoch_y})
                
            epoch_test_loss = epoch_test_loss/len(X_test)
            test_loss_values.append(epoch_test_loss)
            print('Epoch', epoch+1, ' out of ',num_epochs,', train loss:',epoch_train_loss,', test loss:',epoch_test_loss)
            

        # Calculate accuracy on test dataset
        acc = 0
        for epoch_x, epoch_y in zip(X_test, y_test):
            epoch_x = epoch_x.reshape((1, self.img_size, self.img_size, 3))
            epoch_y = epoch_y.reshape((self.img_size*self.img_size, 1))
            
            acc += sess.run(accuracy, feed_dict={X: epoch_x, y: epoch_y})
        
        acc = acc/len(X_test)
        print("Accuracy : ", acc)
        
        # Saving the weights into a file.
        saver.save(sess, model_save_dir+'.ckpt', global_step = step)

        sess.close()

        return train_loss_values, test_loss_values


    def predict(self, img, model_save_dir):
        """
        Prediction.
        ----------------
        Args:
            img: Tensor, the image to be predicted [1 * height * width, 3]
            model_save_dir: the dir of the trained model to be used for prediction.
        
        Returns:
            logits_value: Tensor, predicted annotated image flattened [1, img_size, img_size]
        """
        tf.reset_default_graph()
        
        X = tf.placeholder('float32', [1, self.img_size, self.img_size, 3], name='input_image')        

        logits = self.build(X)
        predictions = tf.round(tf.nn.sigmoid(logits))

        predicted_img = tf.reshape(tf.argmax(logits, axis = 1), [1, self.img_size, self.img_size])
        
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        sess = tf.Session()

        sess.run(init_op)

        saver = tf.train.Saver()

        saver.restore(sess, model_save_dir)

        prediction_value = sess.run(predictions, feed_dict={X: img})
        
        sess.close()
        return prediction_value