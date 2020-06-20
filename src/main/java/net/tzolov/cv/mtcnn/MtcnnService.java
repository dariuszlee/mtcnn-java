/*
 * Copyright 2018 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package net.tzolov.cv.mtcnn;

import static net.tzolov.cv.mtcnn.MtcnnUtil.CHANNEL_COUNT;
import static net.tzolov.cv.mtcnn.MtcnnUtil.C_ORDERING;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.imageio.ImageIO;
import org.apache.commons.io.IOUtils;
import org.bytedeco.javacpp.opencv_core;
import org.bytedeco.javacpp.opencv_imgproc;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.SpecifiedIndex;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.tensorflow.conversion.graphrunner.GraphRunner;
import org.springframework.core.io.DefaultResourceLoader;
import org.springframework.util.Assert;

/** @author Christian Tzolov */
public class MtcnnService {
  public static final String TF_PNET_MODEL_URI = "classpath:/model2/pnet_graph.proto";
  public static final String TF_RNET_MODEL_URI = "classpath:/model2/rnet_graph.proto";
  public static final String TF_ONET_MODEL_URI = "classpath:/model2/onet_graph.proto";

  public static final String MXNET_MODEL_CLASSPATH = "classpath:/mxnet_model";
  public final String MXNET_MODEL_FOLDER;

  private final Java2DNativeImageLoader imageLoader;

  private final Map<Double, MxNetLoader> proposeNetGraphRunnerMxnet;
  private final MxNetLoader refineNetGraphRunnerMxnet;
  private final MxNetLoader outputNetGraphRunnerMxnet;
  private final GraphRunner proposeNetGraphRunner;
  private final GraphRunner refineNetGraphRunner;
  private final GraphRunner outputNetGraphRunner;


  private final int minFaceSize;
  private final double scaleFactor;
  private final double[] stepsThreshold;
  private final int imageWidth;
  private final int imageHeight;
  private final List<Double> scales;

  public MtcnnService(
      int minFaceSize, double scaleFactor, double[] stepsThreshold, int imageWidth, int imageHeight)
      throws IOException {
    this.minFaceSize = minFaceSize;
    this.scaleFactor = scaleFactor;
    this.stepsThreshold = stepsThreshold;

    this.imageLoader = new Java2DNativeImageLoader();

    this.proposeNetGraphRunner = this.createGraphRunner(TF_PNET_MODEL_URI, "pnet/input");
    this.refineNetGraphRunner = this.createGraphRunner(TF_RNET_MODEL_URI, "rnet/input");
    this.outputNetGraphRunner = this.createGraphRunner(TF_ONET_MODEL_URI, "onet/input");

    // End of old constructor
    this.imageWidth = imageWidth;
    this.imageHeight = imageHeight;
    this.scales =
        MtcnnUtil.computeScalePyramid(
            this.imageHeight, this.imageWidth, this.minFaceSize, this.scaleFactor);

    this.MXNET_MODEL_FOLDER =
        new DefaultResourceLoader().getResource(MXNET_MODEL_CLASSPATH).getFile().getAbsolutePath();

    String MXNET_PNET_MODEL_URI = MXNET_MODEL_FOLDER + "/det1";
    String MXNET_RNET_MODEL_URI = MXNET_MODEL_FOLDER + "/det2";
    String MXNET_ONET_MODEL_URI = MXNET_MODEL_FOLDER + "/det3";

    this.proposeNetGraphRunnerMxnet = new HashMap<>();
    for (Double scale : this.scales) {
      int newWidth = (int) Math.ceil(imageWidth * scale);
      int newHeight = (int) Math.ceil(imageHeight * scale);
      MxNetLoader scaledMxNetLoader = this.createGraphRunnerMxnet(MXNET_PNET_MODEL_URI, new int[]{1, 3, newWidth, newHeight});
      proposeNetGraphRunnerMxnet.put(scale, scaledMxNetLoader);
    }

    this.refineNetGraphRunnerMxnet = this.createGraphRunnerMxnet(MXNET_RNET_MODEL_URI, new int[]{1,3, 24, 24});
    this.outputNetGraphRunnerMxnet = this.createGraphRunnerMxnet(MXNET_ONET_MODEL_URI, new int[]{1, 3, 48, 48});
  }

  private MxNetLoader createGraphRunnerMxnet(String mxnetModelUri) {
    return new MxNetLoader(new int[] {0}, mxnetModelUri);
  }

  private MxNetLoader createGraphRunnerMxnet(String mxnetModelUri, int[] sizes) {
    return new MxNetLoader(sizes, mxnetModelUri);
  }

  private GraphRunner createGraphRunner(String tensorflowModelUri, String inputLabel) {
    try {
      return new GraphRunner(
          IOUtils.toByteArray(
              new DefaultResourceLoader().getResource(tensorflowModelUri).getInputStream()),
          Arrays.asList(inputLabel));
      // ConfigProto.getDefaultInstance());
    } catch (IOException e) {
      throw new IllegalStateException(
          String.format(
              "Failed to load TF model [%s] and input [%s]:", tensorflowModelUri, inputLabel),
          e);
    }
  }

  /**
   * Detects faces in an image, and returns bounding boxes and points for them.
   *
   * @param imageUri Uri of the image to detect
   * @return Array of face bounding boxes found in the image
   * @throws IOException Incorrect image Uri.
   */
  public FaceAnnotation[] faceDetection(String imageUri) throws IOException {
    // [ 3 x H x W ]
    INDArray image =
        this.imageLoader
            .asMatrix(new DefaultResourceLoader().getResource(imageUri).getInputStream())
            .get(point(0), interval(0, 3), all(), all())
            .dup();
    // .get(point(0), all(), all(), all()).dup();
    return faceDetection(image);
  }

  public FaceAnnotation[] faceDetection(BufferedImage bImage) throws IOException {
    INDArray ndImage3HW =
        this.imageLoader.asMatrix(bImage).get(point(0), interval(0, 3), all(), all());
    // INDArray ndImage3HW = this.imageLoader.asMatrix(bImage).get(point(0), all(), all(), all());
    return faceDetection(ndImage3HW);
  }

  public FaceAnnotation[] faceDetection(byte[] byteImage, int h, int w) throws IOException {
    INDArray ndImage3HW =
        Nd4j.create(MtcnnUtil.imageByteToFloatArray(byteImage))
            .reshape(new int[] {h, w, 3})
            .permutei(2, 0, 1);
    return faceDetection(ndImage3HW);
  }

  /**
   * Detects faces for byte encoded input images. Supports only byte arrays exported from with their
   * image formats e.g. ImageIO.write(bufferImage, format) or MtcnnUtil.toByteArray(bi2, "png")
   *
   * @param byteImage Input image encoded in bytes along with its image format spec.
   * @return Array of face bounding boxes found in the image
   * @throws IOException Incorrect image Uri.
   */
  public FaceAnnotation[] faceDetection(byte[] byteImage) throws IOException {
    ByteArrayInputStream is = new ByteArrayInputStream(byteImage);
    BufferedImage bufferedImage = ImageIO.read(is);
    return faceDetection(bufferedImage);
  }

  /**
   * Detects faces in an image, and returns bounding boxes and points for them.
   *
   * @param image3HW image to detect the faces in. Expected dimensions [ 3 x H x W ]
   * @return Array of face bounding boxes found in the image
   */
  public FaceAnnotation[] faceDetection(INDArray image3HW) throws IOException {

    INDArray[] outputStageResult = this.rawFaceDetection(image3HW);

    // Convert result into Bounding Box array
    INDArray totalBoxes = outputStageResult[0];
    INDArray points = outputStageResult[1];
    if (!totalBoxes.isEmpty() && totalBoxes.size(0) > 1) {
      points = points.transpose();
    }

    return MtcnnUtil.toFaceAnnotation(totalBoxes, points);
  }

  public INDArray[] faceAlignment(
      INDArray image, FaceAnnotation[] bboxes, int margin, int alignedImageSize, boolean preWhiten)
      throws IOException {
    INDArray[] alignments = new INDArray[bboxes.length];
    for (int i = 0; i < bboxes.length; i++) {
      alignments[i] = this.faceAlignment(image, bboxes[i], margin, alignedImageSize, preWhiten);
    }
    return alignments;
  }

  public INDArray faceAlignment(
      INDArray image,
      FaceAnnotation faceAnnotation,
      int margin,
      int alignedImageSize,
      boolean preWhiten)
      throws IOException {
    FaceAnnotation.BoundingBox bbox = faceAnnotation.getBoundingBox();
    int x = bbox.getX();
    int y = bbox.getY();
    int w = bbox.getW();
    int h = bbox.getH();

    int y1 = Math.max(y - (margin / 2), 0);
    int x1 = Math.max(x - (margin / 2), 0);
    int y2 = Math.min((y + h) + margin / 2, (int) image.shape()[1]);
    int x2 = Math.min((x + w) + margin / 2, (int) image.shape()[2]);

    INDArray croppedImage = MtcnnUtil.crop(image, x1, x2, y1, y2);

    croppedImage =
        this.resize(croppedImage, new opencv_core.Size(alignedImageSize, alignedImageSize)); // W, H

    if (preWhiten) {
      croppedImage = MtcnnUtil.preWhiten(croppedImage);
    }

    return croppedImage;
  }

  /**
   * Detect faces and related points.
   *
   * @param image3HW input image with dimensions [C x H x W] (e.g. channels first)
   * @return Two INDArray elements representing the Total Boxes found and the related points.
   * @throws IOException
   */
  public INDArray[] rawFaceDetection(INDArray image3HW) throws IOException {

    WorkspaceConfiguration initialConfig =
        WorkspaceConfiguration.builder()
            .initialSize(10 * 1024L * 1024L)
            .policyAllocation(AllocationPolicy.STRICT)
            .policyLearning(LearningPolicy.NONE)
            .build();

    try (MemoryWorkspace ws =
        Nd4j.getWorkspaceManager().getAndActivateWorkspace(initialConfig, "SOME_ID")) {

      Assert.isTrue(
          image3HW.rank() == 3,
          "The input image is expected to have [0, Channels, Width, Height] dimensions");
      Assert.isTrue(
          image3HW.shape()[0] == 3,
          "The input image is expected to have channel count at dimension 0");

      // Compute the scale pyramid
      int height = (int) image3HW.size(1);
      int width = (int) image3HW.size(2);

      // Stage One
      Object[] stageOneResult = this.preparationStage(image3HW, scales);

      // Reorder image dimensions from [3,H,W] to [H,W,3]
      image3HW = image3HW.permute(1, 2, 0);

      // Stage Two
      INDArray totalBoxes =
          this.refinementStage(
              image3HW, (INDArray) stageOneResult[0], (MtcnnUtil.PadResult) stageOneResult[1]);

      // Stage Three
      INDArray[] stageThreeResult = this.outputStage(image3HW, totalBoxes);

      return stageThreeResult;
    }
  }

  /**
   * STAGE 1
   *
   * @param image3HW
   * @param scales
   * @return
   * @throws IOException
   */
  private Object[] preparationStage(INDArray image3HW, List<Double> scales) throws IOException {

    INDArray totalBoxes = Nd4j.empty();
    MtcnnUtil.PadResult padResult = null;

    double imageHeight = image3HW.size(1);
    double imageWidth = image3HW.size(2);

    for (Double scale : this.scales) {

      int newWidth = (int) Math.ceil(imageWidth * scale);
      int newHeight = (int) Math.ceil(imageHeight * scale);

      MxNetLoader currentPropose = proposeNetGraphRunnerMxnet.get(scale);
      INDArray tempImg =
          resize(image3HW, new opencv_core.Size(newWidth, newHeight)).permute(0, 1, 3, 2).dup();
      tempImg = tempImg.sub(127.5).mul(0.0078125);

      List<INDArray> resultList = currentPropose.runPropose(tempImg);
      INDArray out0 = resultList.get(0);
      INDArray out1 = resultList.get(1);

      // Map<String, INDArray> resultMap =
      //     this.proposeNetGraphRunner.run(Collections.singletonMap("pnet/input", image0WH3));
      // INDArray out0_n = resultMap.get("pnet/conv4-2/BiasAdd");
      // System.out.println("DARIUS SHAPE0a " + out0_n.shapeInfoToString());
      // INDArray out1_n = resultMap.get("pnet/prob1");

      INDArray boxes =
          MtcnnUtil.generateBoundingBox(
              out1.get(point(0), all(), all(), point(1)),
              out0.get(point(0), all(), all(), all()),
              scale,
              this.stepsThreshold[0])[0];

      if (!boxes.isEmpty()) {
        INDArray pick =
            MtcnnUtil.nonMaxSuppression(boxes, 0.5, MtcnnUtil.NonMaxSuppressionType.Union);
        if (boxes.length() > 0 && pick.length() > 0 && !pick.isEmpty()) {
          boxes = boxes.get(new SpecifiedIndex(pick.toLongVector()), all());
          if (totalBoxes.isEmpty()) {
            totalBoxes = boxes;
          } else {
            totalBoxes = MtcnnUtil.append(totalBoxes, boxes, 0);
          }
        }
      }
    }

    long numBoxes = totalBoxes.isEmpty() ? 0 : totalBoxes.shape()[0];
    if (numBoxes > 0) {
      INDArray pick =
          MtcnnUtil.nonMaxSuppression(totalBoxes, 0.7, MtcnnUtil.NonMaxSuppressionType.Union);
      totalBoxes = totalBoxes.get(new SpecifiedIndex(pick.toLongVector()), all());

      // regw = total_boxes[:, 2] - total_boxes[:, 0]
      // regh = total_boxes[:, 3] - total_boxes[:, 1]
      INDArray x2 = totalBoxes.get(all(), point(2));
      INDArray x1 = totalBoxes.get(all(), point(0));
      INDArray y2 = totalBoxes.get(all(), point(3));
      INDArray y1 = totalBoxes.get(all(), point(1));

      INDArray regw = x2.sub(x1);
      INDArray regh = y2.sub(y1);

      // qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
      // qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
      // qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
      // qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh
      INDArray qq1 = x1.add(totalBoxes.get(all(), point(5)).mul(regw));
      INDArray qq2 = y1.add(totalBoxes.get(all(), point(6)).mul(regh));
      INDArray qq3 = x2.add(totalBoxes.get(all(), point(7)).mul(regw));
      INDArray qq4 = y2.add(totalBoxes.get(all(), point(8)).mul(regh));

      // total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4, total_boxes[:, 4]]))
      totalBoxes = Nd4j.hstack(qq1, qq2, qq3, qq4, totalBoxes.get(all(), point(4)));

      // total_boxes = self.__rerec(total_boxes.copy())
      totalBoxes = MtcnnUtil.rerec(totalBoxes, true);

      padResult = MtcnnUtil.pad(totalBoxes, (int) imageWidth, (int) imageHeight);
    }

    return new Object[] {totalBoxes, padResult};
  }

  /**
   * STAGE 2
   *
   * @param image
   * @param totalBoxes
   * @param padResult
   * @return
   * @throws IOException
   */
  private INDArray refinementStage(
      INDArray image, INDArray totalBoxes, MtcnnUtil.PadResult padResult) throws IOException {

    int numBoxes = totalBoxes.isEmpty() ? 0 : (int) totalBoxes.shape()[0];
    if (numBoxes == 0) {
      return totalBoxes;
    }

    INDArray tempImg1 = computeTempImage(image, numBoxes, padResult, 24);
    INDArray refineTemp = tempImg1.permute(0, 3, 1, 2);
    List<INDArray> resultList = refineNetGraphRunnerMxnet.runRefine(refineTemp);
    INDArray out0 = resultList.get(0);
    INDArray out1 = resultList.get(1);

    // Map<String, INDArray> resultMap =
    //     this.refineNetGraphRunner.run(Collections.singletonMap("rnet/input", tempImg1));
    // INDArray out0 = resultMap.get("rnet/conv5-2/conv5-2");
    // INDArray out1 = resultMap.get("rnet/prob1");

    //  score = out1[1, :]
    INDArray score = out1.get(all(), point(1)).transposei();

    INDArray ipass = MtcnnUtil.getIndexWhereVector(score.transpose(), s -> s > stepsThreshold[1]);

    if (ipass.isEmpty()) {
      totalBoxes = Nd4j.empty();
      return totalBoxes;
    }

    INDArray b1 = totalBoxes.get(new SpecifiedIndex(ipass.toLongVector()), interval(0, 4));
    INDArray b2 =
        ipass.isScalar() ? score.get(ipass).reshape(1, 1) : Nd4j.expandDims(score.get(ipass), 1);
    totalBoxes = Nd4j.hstack(b1, b2);

    INDArray mv = out0.get(new SpecifiedIndex(ipass.toLongVector()), all()).transposei();

    if (!totalBoxes.isEmpty() && totalBoxes.shape()[0] > 0) {
      INDArray pick =
          MtcnnUtil.nonMaxSuppression(totalBoxes.dup(), 0.7, MtcnnUtil.NonMaxSuppressionType.Union)
              .transpose();

      totalBoxes = totalBoxes.get(new SpecifiedIndex(pick.toLongVector()), all());

      totalBoxes =
          MtcnnUtil.bbreg(
              totalBoxes, mv.get(all(), new SpecifiedIndex(pick.toLongVector())).transpose());

      totalBoxes = MtcnnUtil.rerec(totalBoxes, false);
    }

    return totalBoxes;
  }

  /**
   * STAGE 3
   *
   * @param image
   * @param totalBoxes
   * @return
   * @throws IOException
   */
  private INDArray[] outputStage(INDArray image, INDArray totalBoxes) throws IOException {

    int numBoxes = totalBoxes.isEmpty() ? 0 : (int) totalBoxes.shape()[0];
    if (numBoxes == 0) {
      return new INDArray[] {totalBoxes, Nd4j.empty()};
    }

    totalBoxes = Transforms.floor(totalBoxes);

    MtcnnUtil.PadResult padResult =
        MtcnnUtil.pad(totalBoxes, (int) image.shape()[1], (int) image.shape()[0]);

    INDArray tempImg1 = computeTempImage(image, numBoxes, padResult, 48);
    System.out.println("Temp img1" + tempImg1.shapeInfoToString());
    INDArray outputTemp = tempImg1.permute(0, 3, 2, 1);

    List<INDArray> resultList = outputNetGraphRunnerMxnet.runOutput(outputTemp);
    INDArray out0 = resultList.get(1);
    INDArray out1 = resultList.get(0);
    INDArray out2 = resultList.get(2);

    Map<String, INDArray> resultMap =
        this.outputNetGraphRunner.run(Collections.singletonMap("onet/input", tempImg1));

    // INDArray out0 = resultMap.get("onet/conv6-2/conv6-2");
    // INDArray out1 = resultMap.get("onet/conv6-3/conv6-3");
    // INDArray out2 = resultMap.get("onet/prob1");
    System.out.println("SHAPE0: "+ out0.shapeInfoToString());
    System.out.println("SHAPE1: "+ out1.shapeInfoToString());
    System.out.println("SHAPE2: "+ out2.shapeInfoToString());

    INDArray score = out2.get(all(), point(1)).transposei();
    System.out.println("SCORES: " + score);

    INDArray points = out1;

    INDArray ipass = MtcnnUtil.getIndexWhereVector(score.transpose(), s -> s > stepsThreshold[2]);
    System.out.println("IPASS: " + ipass);

    if (ipass.isEmpty()) {
      return new INDArray[] {Nd4j.empty(), Nd4j.empty()};
    }

    points = points.get(new SpecifiedIndex(ipass.toLongVector()), all()).transposei();

    INDArray b1 = totalBoxes.get(new SpecifiedIndex(ipass.toLongVector()), interval(0, 4)).dup();
    INDArray b2 =
        ipass.isScalar()
            ? score.get(ipass).reshape(1, 1).dup()
            : Nd4j.expandDims(score.get(ipass).dup(), 1);
    totalBoxes = Nd4j.hstack(b1, b2);
    System.out.println("TOTAL BOXES " + totalBoxes);

    INDArray mv = out0.get(new SpecifiedIndex(ipass.toLongVector()), all()).transposei();

    INDArray w =
        totalBoxes.get(all(), point(2)).dup().subi(totalBoxes.get(all(), point(0))).addi(1);
    INDArray h =
        totalBoxes.get(all(), point(3)).dup().subi(totalBoxes.get(all(), point(1))).addi(1);

    points.put(
        new INDArrayIndex[] {interval(0, 5), all()},
        Nd4j.repeat(w, 5)
            .muli(points.get(interval(0, 5), all()))
            .addi(Nd4j.repeat(totalBoxes.get(all(), point(0)), 5))
            .subi(1));

    points.put(
        new INDArrayIndex[] {interval(5, 10), all()},
        Nd4j.repeat(h, 5)
            .muli(points.get(interval(5, 10), all()))
            .addi(Nd4j.repeat(totalBoxes.get(all(), point(1)), 5))
            .subi(1));

    if (totalBoxes.shape()[0] > 0) {
      totalBoxes = MtcnnUtil.bbreg(totalBoxes.dup(), mv.transpose());

      INDArray pick =
          MtcnnUtil.nonMaxSuppression(totalBoxes.dup(), 0.7, MtcnnUtil.NonMaxSuppressionType.Min)
              .transpose();

      totalBoxes = totalBoxes.get(new SpecifiedIndex(pick.toLongVector()), all());

      points = points.get(all(), new SpecifiedIndex(pick.toLongVector()));
    }

    return new INDArray[] {totalBoxes, points};
  }

  private INDArray computeTempImage(
      INDArray image, int numBoxes, MtcnnUtil.PadResult padResult, int size) throws IOException {

    //  tempimg = np.zeros(shape=(size, size, 3, num_boxes))
    INDArray tempImg = Nd4j.zeros(new int[] {size, size, CHANNEL_COUNT, numBoxes}, C_ORDERING);

    opencv_core.Size newSize = new opencv_core.Size(size, size);

    for (int k = 0; k < numBoxes; k++) {
      INDArray tmp =
          Nd4j.zeros(
              new int[] {
                padResult.getTmph().getInt(k), padResult.getTmpw().getInt(k), CHANNEL_COUNT
              },
              C_ORDERING);

      tmp.put(
          new INDArrayIndex[] {
            interval(padResult.getDy().getInt(k) - 1, padResult.getEdy().getInt(k)),
            interval(padResult.getDx().getInt(k) - 1, padResult.getEdx().getInt(k)),
            all()
          },
          image.get(
              interval(padResult.getY().getInt(k) - 1, padResult.getEy().getInt(k)),
              interval(padResult.getX().getInt(k) - 1, padResult.getEx().getInt(k)),
              all()));

      if ((tmp.shape()[0] > 0 && tmp.shape()[1] > 0)
          || (tmp.shape()[0] == 0 && tmp.shape()[1] == 0)) {

        INDArray resizedImage =
            resize(tmp.permutei(2, 0, 1).dup(), newSize)
                .get(point(0), all(), all(), all())
                .permutei(1, 2, 0)
                .dup();

        tempImg.put(new INDArrayIndex[] {all(), all(), all(), point(k)}, resizedImage);
      } else {
        return Nd4j.empty();
      }
    }

    // tempimg = (tempimg - 127.5) * 0.0078125
    tempImg = tempImg.subi(127.5).muli(0.0078125);

    // tempimg1 = np.transpose(tempimg, (3, 1, 0, 2))
    INDArray tempImg1 = tempImg.permutei(3, 1, 0, 2).dup();

    return tempImg1;
  }

  /**
   * Resize an {@link INDArray} encoded image.
   *
   * @param imageCHW Image to resize. Expects [CHANNEL, HEIGHT, WIDTH] dimensions.
   * @param newSizeWH new image size (w,h)
   * @return Returns {@link INDArray} resized image with following dimensions [BATCH, WIDTH, HEIGHT,
   *     CHANNEL]
   * @throws IOException
   */
  public INDArray resize(INDArray imageCHW, opencv_core.Size newSizeWH) throws IOException {
    Assert.isTrue(
        imageCHW.size(0) == CHANNEL_COUNT,
        "Input image is expected to have the [3, W, H] dimensions");
    // Mat expects [C, H, W] dimensions
    opencv_core.Mat mat = imageLoader.asMat(imageCHW);
    opencv_imgproc.resize(mat, mat, newSizeWH, 0, 0, opencv_imgproc.CV_INTER_AREA);

    // Mat mat = imageLoader.asMat(imageCHW);
    // Imgproc.resize(mat, mat, newSizeWH, 0, 0, Imgproc.CV_INTER_AREA);
    // [0, W, H, 3]
    return imageLoader.asMatrix(mat);
  }
}