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

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;

import javax.imageio.ImageIO;
import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;
import org.datavec.image.loader.Java2DNativeImageLoader;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.bytedeco.javacpp.opencv_core;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.springframework.core.io.DefaultResourceLoader;

import static org.hamcrest.CoreMatchers.equalTo;
import static org.junit.Assert.assertThat;

/**
 * @author Christian Tzolov
 */
public class MtcnnServiceTest {


	private MtcnnService mtcnnService;
    private Java2DNativeImageLoader imageLoader;

	@Before
	public void before() {
		mtcnnService = new MtcnnService(20, 0.709, new double[] { 0.6, 0.7, 0.7 });
		this.imageLoader = new Java2DNativeImageLoader();
	}

	@Test
	public void testSingeFace() throws IOException {
		try (InputStream imageInputStream = new DefaultResourceLoader()
				.getResource("classpath:/Anthony_Hopkins_0002.jpg").getInputStream()) {
			// 1. Load the input image (you can use http:/, file:/ or classpath:/ URIs to resolve the input image
			BufferedImage inputImage = ImageIO.read(imageInputStream);
            // INDArray ndImage3HW = mtcnnService.resize(this.imageLoader.asMatrix(inputImage).get(point(0), interval(0,3), all(), all()), new opencv_core.Size(480, 640));
            INDArray ndImage3HW = this.imageLoader.asMatrix(inputImage).get(point(0), interval(0,3), all(), all());
            inputImage = this.imageLoader.asBufferedImage(ndImage3HW);
			// 2. Run face detection
			FaceAnnotation[] faceAnnotations = mtcnnService.faceDetection(ndImage3HW);


			// 3. Augment the input image with the detected faces
			BufferedImage annotatedImage = MtcnnUtil.drawFaceAnnotations(inputImage, faceAnnotations);
			// 4. Store face-annotated image
			ImageIO.write(annotatedImage, "png", new File("./AnnotatedImage.png"));
            assertThat(toJson(faceAnnotations), equalTo("[{\"bbox\":{\"x\":75,\"y\":67,\"w\":95,\"h\":120}," +
                    "\"confidence\":0.9994938373565674," +
                    "\"landmarks\":[" +
                    "{\"type\":\"LEFT_EYE\",\"position\":{\"x\":101,\"y\":113}}," +
                    "{\"type\":\"RIGHT_EYE\",\"position\":{\"x\":147,\"y\":113}}," +
                    "{\"type\":\"NOSE\",\"position\":{\"x\":124,\"y\":136}}," +
                    "{\"type\":\"MOUTH_LEFT\",\"position\":{\"x\":105,\"y\":160}}," +
                    "{\"type\":\"MOUTH_RIGHT\",\"position\":{\"x\":146,\"y\":160}}]}]"));
            }
	}

	@Test
	public void testFailToDetectFace() throws IOException {
		FaceAnnotation[] faceAnnotations = mtcnnService.faceDetection("classpath:/broken.png");
		assertThat(toJson(faceAnnotations), equalTo("[]"));
	}

	@Test
	public void testMultiFaces() throws IOException {
		FaceAnnotation[] faceAnnotations = mtcnnService.faceDetection("classpath:/VikiMaxiAdi.jpg");
		assertThat(toJson(faceAnnotations), equalTo("[{\"bbox\":{\"x\":102,\"y\":152,\"w\":70,\"h\":86}," +
				"\"confidence\":0.9999865293502808," +
				"\"landmarks\":[" +
				"{\"type\":\"LEFT_EYE\",\"position\":{\"x\":122,\"y\":189}}," +
				"{\"type\":\"RIGHT_EYE\",\"position\":{\"x\":154,\"y\":190}}," +
				"{\"type\":\"NOSE\",\"position\":{\"x\":136,\"y\":203}}," +
				"{\"type\":\"MOUTH_LEFT\",\"position\":{\"x\":122,\"y\":219}}," +
				"{\"type\":\"MOUTH_RIGHT\",\"position\":{\"x\":151,\"y\":220}}]}," +
				"{\"bbox\":{\"x\":332,\"y\":94,\"w\":57,\"h\":69}," +
				"\"confidence\":0.9992565512657166," +
				"\"landmarks\":[" +
				"{\"type\":\"LEFT_EYE\",\"position\":{\"x\":346,\"y\":120}}," +
				"{\"type\":\"RIGHT_EYE\",\"position\":{\"x\":373,\"y\":121}}," +
				"{\"type\":\"NOSE\",\"position\":{\"x\":357,\"y\":134}}," +
				"{\"type\":\"MOUTH_LEFT\",\"position\":{\"x\":346,\"y\":147}}," +
				"{\"type\":\"MOUTH_RIGHT\",\"position\":{\"x\":370,\"y\":148}}]}]"));
	}

	private String toJson(FaceAnnotation[] faceAnnotations) throws JsonProcessingException {
		return new ObjectMapper().writeValueAsString(faceAnnotations);
	}
}
