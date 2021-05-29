from absl import app, flags, logging
from absl.flags import FLAGS
import os
import shutil
import tensorflow as tf
from core.yolov4 import YOLO, decode, compute_loss, decode_train, filter_boxes
from core.dataset import Dataset
from core.config import cfg
import numpy as np
from core import utils
from core.utils import freeze_all, unfreeze_all, image_preprocess, draw_bbox
import cv2
from PIL import Image
from google.colab.patches import cv2_imshow

flags.DEFINE_string('model', 'yolov4', 'yolov4, yolov3')
flags.DEFINE_string('weights', '/content/tensorflow-yolov4-tflite/checkpoints_m/yolov4', 'pretrained weights')
flags.DEFINE_boolean('tiny', False, 'yolo or yolo-tiny')
flags.DEFINE_string('output', '/content/tensorflow-yolov4-tflite/result.png', 'path to output image')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.25, 'score threshold')

def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # if len(physical_devices) > 0:
    #     tf.config.experimental.set_memory_growth(physical_devices[0], True)

    trainset = Dataset(FLAGS, is_training=True)
    testset = Dataset(FLAGS, is_training=False)
    logdir = "./data/log"
    isfreeze = False
    steps_per_epoch = len(trainset)
    first_stage_epochs = 1 #cfg.TRAIN.FISRT_STAGE_EPOCHS
    second_stage_epochs = 0 #cfg.TRAIN.SECOND_STAGE_EPOCHS
    global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
    warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
    total_steps = (first_stage_epochs + second_stage_epochs) * steps_per_epoch
    # train_steps = (first_stage_epochs + second_stage_epochs) * steps_per_period

    input_layer = tf.keras.layers.Input([cfg.TRAIN.INPUT_SIZE, cfg.TRAIN.INPUT_SIZE, 3])
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
    IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESH

    freeze_layers = utils.load_freeze_layer(FLAGS.model, FLAGS.tiny)

    feature_maps = YOLO(input_layer, NUM_CLASS, FLAGS.model, FLAGS.tiny)
    if FLAGS.tiny:
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            else:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)
    else:
        bbox_tensors = []
        for i, fm in enumerate(feature_maps):
            if i == 0:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            elif i == 1:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            else:
                bbox_tensor = decode_train(fm, cfg.TRAIN.INPUT_SIZE // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE)
            bbox_tensors.append(fm)
            bbox_tensors.append(bbox_tensor)

    model = tf.keras.Model(input_layer, bbox_tensors)
    #model.summary()

    if FLAGS.weights == None:
        print("Training from scratch")
    else:
        if FLAGS.weights.split(".")[len(FLAGS.weights.split(".")) - 1] == "weights":
            utils.load_weights(model, FLAGS.weights, FLAGS.model, FLAGS.tiny)
        else:
            model.load_weights(FLAGS.weights)
        print('Restoring weights from: %s ... ' % FLAGS.weights)
    
    for image_data, target in testset: #batch_size in cfg file = 1
      image_path = '/content/tensorflow-yolov4-tflite/Combined_F/images/Trip017Day119-10-05Image14add_1.png'

      original_image = cv2.imread(image_path)
      #original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

      #image_data = image_preprocess(np.copy(original_image), [416, 416])
      image_data = cv2.resize(original_image, (416, 416))
      image_data = image_data / 255.
      # image_data = image_data[np.newaxis, ...].astype(np.float32)

      images_data = []
      for i in range(1):
        images_data.append(image_data)
      images_data = np.asarray(images_data).astype(np.float32)
      pred_result = model(images_data, training=False)
      #batch_data = tf.constant(images_data)
      #pred_bbox = infer(batch_data)
      pred_result_1 = tf.reshape(pred_result[1],(tf.shape(pred_result[1])[0], -1, 6))
      pred_result_2 = tf.reshape(pred_result[3],(tf.shape(pred_result[3])[0], -1, 6))
      pred_result_3 = tf.reshape(pred_result[5],(tf.shape(pred_result[5])[0], -1, 6))
      value = tf.concat([pred_result_1,pred_result_2,pred_result_3],axis=1)
      print(tf.shape(value))
      print(tf.math.reduce_sum(value))
      boxes = value[:, :, 0:4]
      pred_conf = tf.math.multiply(tf.reshape(value[:, :, 4], (1,-1,1)),value[:, :, 5:])
      print('boxes:', tf.shape(boxes))
      print('pred_conf:', tf.shape(pred_conf))
      print(boxes.numpy())

      boxes, pred_conf = filter_boxes(boxes, pred_conf, score_threshold=FLAGS.score, input_shape=tf.constant([416, 416]))


      boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=FLAGS.iou,
        score_threshold=FLAGS.score,
        clip_boxes=True)
      
      pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
      print(pred_bbox)
      print(original_image.shape)
      image = draw_bbox(original_image, pred_bbox)
      print(image.shape)
    # image = utils.draw_bbox(image_data*255, pred_bbox)
      image = Image.fromarray(image.astype(np.uint8))
      image.show()
      #cv2_imshow()
      image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
      cv2.imwrite(FLAGS.output, image)
      break
    # optimizer = tf.keras.optimizers.Adam()
    # if os.path.exists(logdir): shutil.rmtree(logdir)
    # writer = tf.summary.create_file_writer(logdir)

    # # define training step function
    # # @tf.function
    # def train_step(image_data, target):
    #     with tf.GradientTape() as tape:
    #         pred_result = model(image_data, training=True)
    #         giou_loss = conf_loss = prob_loss = 0

    #         # optimizing process
    #         for i in range(len(freeze_layers)):
    #             conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
    #             loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
    #             giou_loss += loss_items[0]
    #             conf_loss += loss_items[1]
    #             prob_loss += loss_items[2]

    #         total_loss = giou_loss + conf_loss + prob_loss

    #         gradients = tape.gradient(total_loss, model.trainable_variables)
    #         optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    #         tf.print("=> STEP %4d/%4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
    #                  "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, total_steps, optimizer.lr.numpy(),
    #                                                            giou_loss, conf_loss,
    #                                                            prob_loss, total_loss))
    #         # update learning rate
    #         global_steps.assign_add(1)
    #         if global_steps < warmup_steps:
    #             lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT
    #         else:
    #             lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
    #                 (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
    #             )
    #         optimizer.lr.assign(lr.numpy())

    #         # writing summary data
    #         with writer.as_default():
    #             tf.summary.scalar("lr", optimizer.lr, step=global_steps)
    #             tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
    #             tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
    #             tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
    #             tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
    #         writer.flush()
    # def test_step(image_data, target):
    #     with tf.GradientTape() as tape:
    #         pred_result = model(image_data, training=True)
    #         giou_loss = conf_loss = prob_loss = 0

    #         # optimizing process
    #         for i in range(len(freeze_layers)):
    #             conv, pred = pred_result[i * 2], pred_result[i * 2 + 1]
    #             loss_items = compute_loss(pred, conv, target[i][0], target[i][1], STRIDES=STRIDES, NUM_CLASS=NUM_CLASS, IOU_LOSS_THRESH=IOU_LOSS_THRESH, i=i)
    #             giou_loss += loss_items[0]
    #             conf_loss += loss_items[1]
    #             prob_loss += loss_items[2]

    #         total_loss = giou_loss + conf_loss + prob_loss

    #         tf.print("=> TEST STEP %4d   giou_loss: %4.2f   conf_loss: %4.2f   "
    #                  "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps, giou_loss, conf_loss,
    #                                                            prob_loss, total_loss))

    # for epoch in range(first_stage_epochs + second_stage_epochs):
    #     if epoch < first_stage_epochs:
    #         if not isfreeze:
    #             isfreeze = True
    #             for name in freeze_layers:
    #                 freeze = model.get_layer(name)
    #                 freeze_all(freeze)
    #     elif epoch >= first_stage_epochs:
    #         if isfreeze:
    #             isfreeze = False
    #             for name in freeze_layers:
    #                 freeze = model.get_layer(name)
    #                 unfreeze_all(freeze)
    #     for image_data, target in trainset:
    #         train_step(image_data, target)
    #     for image_data, target in testset:
    #         test_step(image_data, target)
    #     model.save_weights("./checkpoints/yolov4")
    # model.save_weights("/content/tensorflow-yolov4-tflite/checkpoints_m/yolov4")
    # model.save("/content/tensorflow-yolov4-tflite/checkpoints__m/yolov4")

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass