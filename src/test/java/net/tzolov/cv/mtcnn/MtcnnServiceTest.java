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
      String shouldEquals = "[{\"bbox\":{\"x\":71,\"y\":54,\"w\":100,\"h\":135},\"confidence\":0.9999998807907104,\"landmarks\":[{\"type\":\"LEFT_EYE\",\"position\":{\"x\":102,\"y\":112}},{\"type\":\"RIGHT_EYE\",\"position\":{\"x\":149,\"y\":111}},{\"type\":\"NOSE\",\"position\":{\"x\":127,\"y\":138}},{\"type\":\"MOUTH_LEFT\",\"position\":{\"x\":106,\"y\":161}},{\"type\":\"MOUTH_RIGHT\",\"position\":{\"x\":147,\"y\":159}}]}]";
      assertThat(
          toJson(faceAnnotations),
          equalTo(shouldEquals));
      System.out.println("DARIUS: success");
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
      System.out.println("DARIUS: success");
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
      String shouldEquals = "[{\"bbox\":{\"x\":103,\"y\":158,\"w\":67,\"h\":81},\"confidence\":0.9999957084655762,\"landmarks\":[{\"type\":\"LEFT_EYE\",\"position\":{\"x\":120,\"y\":188}},{\"type\":\"RIGHT_EYE\",\"position\":{\"x\":153,\"y\":190}},{\"type\":\"NOSE\",\"position\":{\"x\":135,\"y\":204}},{\"type\":\"MOUTH_LEFT\",\"position\":{\"x\":120,\"y\":219}},{\"type\":\"MOUTH_RIGHT\",\"position\":{\"x\":148,\"y\":221}}]},{\"bbox\":{\"x\":329,\"y\":86,\"w\":62,\"h\":80},\"confidence\":0.9993091821670532,\"landmarks\":[{\"type\":\"LEFT_EYE\",\"position\":{\"x\":345,\"y\":119}},{\"type\":\"RIGHT_EYE\",\"position\":{\"x\":373,\"y\":119}},{\"type\":\"NOSE\",\"position\":{\"x\":357,\"y\":133}},{\"type\":\"MOUTH_LEFT\",\"position\":{\"x\":346,\"y\":150}},{\"type\":\"MOUTH_RIGHT\",\"position\":{\"x\":370,\"y\":150}}]},{\"bbox\":{\"x\":336,\"y\":58,\"w\":57,\"h\":66},\"confidence\":0.09178774058818817,\"landmarks\":[{\"type\":\"LEFT_EYE\",\"position\":{\"x\":359,\"y\":76}},{\"type\":\"RIGHT_EYE\",\"position\":{\"x\":374,\"y\":79}},{\"type\":\"NOSE\",\"position\":{\"x\":362,\"y\":79}},{\"type\":\"MOUTH_LEFT\",\"position\":{\"x\":353,\"y\":92}},{\"type\":\"MOUTH_RIGHT\",\"position\":{\"x\":368,\"y\":94}}]}]";


      assertThat(
          toJson(faceAnnotations),
          equalTo(shouldEquals));
      System.out.println("DARIUS: success");

    }
  }

  private String toJson(FaceAnnotation[] faceAnnotations) throws JsonProcessingException {
    return new ObjectMapper().writeValueAsString(faceAnnotations);
  }
}
