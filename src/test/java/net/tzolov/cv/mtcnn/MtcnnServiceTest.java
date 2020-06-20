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

import static org.hamcrest.CoreMatchers.equalTo;
import static org.junit.Assert.assertThat;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import javax.imageio.ImageIO;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.springframework.core.io.DefaultResourceLoader;

/** @author Christian Tzolov */
public class MtcnnServiceTest {

  private MtcnnService mtcnnService;
  private Java2DNativeImageLoader imageLoader;

  @Before
  public void before() {
    this.imageLoader = new Java2DNativeImageLoader();
  }

  @Test
  public void testSingeFace() throws IOException {
    try (InputStream imageInputStream =
        new DefaultResourceLoader()
            .getResource("classpath:/Anthony_Hopkins_0002.jpg")
            .getInputStream()) {

      BufferedImage inputImage = ImageIO.read(imageInputStream);
      mtcnnService =
          new MtcnnService(
              20,
              0.709,
              new double[] {0.6, 0.7, 0.8},
              inputImage.getWidth(),
              inputImage.getHeight());

      INDArray ndImage3HW =
          this.imageLoader.asMatrix(inputImage).get(point(0), interval(0, 3), all(), all());
      inputImage = this.imageLoader.asBufferedImage(ndImage3HW);
      // 2. Run face detection
      FaceAnnotation[] faceAnnotations = mtcnnService.faceDetection(ndImage3HW);

      // 3. Augment the input image with the detected faces
      BufferedImage annotatedImage = MtcnnUtil.drawFaceAnnotations(inputImage, faceAnnotations);
      // 4. Store face-annotated image
      ImageIO.write(annotatedImage, "png", new File("./AnnotatedImage.png"));
      String shouldEqual = "[{\"bbox\":{\"x\":75,\"y\":65,\"w\":96,\"h\":123},\"confidence\":0.9999769926071167,\"land"
           + "marks\":[{\"type\":\"LEFT_EYE\",\"position\":{\"x\":103,\"y\":113}},{\"type\":\"RIGHT_EYE\",\"position\":"
           + "{\"x\":149,\"y\":114}},{\"type\":\"NOSE\",\"position\":{\"x\":125,\"y\":136}},{\"type\":\"MOUTH_LEFT\",\""
           + "position\":{\"x\":106,\"y\":160}},{\"type\":\"MOUTH_RIGHT\",\"position\":{\"x\":146,\"y\":160}}]}]";
      assertThat(
          toJson(faceAnnotations),
          equalTo(shouldEqual));
    }
  }

  @Test
  public void testFailToDetectFace() throws IOException {
    try (InputStream imageInputStream =
        new DefaultResourceLoader().getResource("classpath:/broken.png").getInputStream()) {
      BufferedImage inputImage = ImageIO.read(imageInputStream);
      mtcnnService =
          new MtcnnService(
              20,
              0.709,
              new double[] {0.6, 0.7, 0.7},
              inputImage.getWidth(),
              inputImage.getHeight());
      FaceAnnotation[] faceAnnotations = mtcnnService.faceDetection(inputImage);
      assertThat(toJson(faceAnnotations), equalTo("[]"));
    }
  }

  @Test
  public void testMultiFaces() throws IOException {
    try (InputStream imageInputStream =
        new DefaultResourceLoader().getResource("classpath:/VikiMaxiAdi.jpg").getInputStream()) {
      BufferedImage inputImage = ImageIO.read(imageInputStream);
      mtcnnService =
          new MtcnnService(
              20,
              0.709,
              new double[] {0.6, 0.7, 0.7},
              inputImage.getWidth(),
              inputImage.getHeight());
      FaceAnnotation[] faceAnnotations = mtcnnService.faceDetection(inputImage);
      String shouldEqual = "[{\"bbox\":{\"x\":331,\"y\":94,\"w\":58,\"h\":70},\"confidence\":0.9999604225158691,\"land"
 + "marks\":[{\"type\":\"LEFT_EYE\",\"position\":{\"x\":346,\"y\":121}},{\"type\":\"RIGHT_EYE\",\"position\":"
 + "{\"x\":373,\"y\":120}},{\"type\":\"NOSE\",\"position\":{\"x\":358,\"y\":135}},{\"type\":\"MOUTH_LEFT\",\""
 + "position\":{\"x\":347,\"y\":147}},{\"type\":\"MOUTH_RIGHT\",\"position\":{\"x\":370,\"y\":147}}]},{\"bbox"
 + "\":{\"x\":102,\"y\":157,\"w\":69,\"h\":83},\"confidence\":0.9997490048408508,\"landmarks\":[{\"type\":\"L"
 + "EFT_EYE\",\"position\":{\"x\":121,\"y\":189}},{\"type\":\"RIGHT_EYE\",\"position\":{\"x\":153,\"y\":189}}"
 + ",{\"type\":\"NOSE\",\"position\":{\"x\":135,\"y\":204}},{\"type\":\"MOUTH_LEFT\",\"position\":{\"x\":122,"
 + "\"y\":218}},{\"type\":\"MOUTH_RIGHT\",\"position\":{\"x\":149,\"y\":220}}]}]";
      assertThat(
          toJson(faceAnnotations),
          equalTo(shouldEqual));

    }
  }

  private String toJson(FaceAnnotation[] faceAnnotations) throws JsonProcessingException {
    return new ObjectMapper().writeValueAsString(faceAnnotations);
  }
}
