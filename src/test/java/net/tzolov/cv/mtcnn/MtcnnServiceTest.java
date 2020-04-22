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

import java.io.IOException;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.Before;
import org.junit.Test;

import static org.hamcrest.CoreMatchers.equalTo;
import static org.junit.Assert.assertThat;

/**
 * @author Christian Tzolov
 */
public class MtcnnServiceTest {


	private MtcnnService mtcnnService;

	@Before
	public void before() {
		mtcnnService = new MtcnnService(20, 0.709, new double[] { 0.6, 0.7, 0.7 });
	}

    @Test
    public void testInvalidPadding() throws IOException {
        String path = "classpath:0001_00_00_01_0.jpg";
		FaceAnnotation[] faceAnnotations = mtcnnService.faceDetection(path);
    }

	@Test
	public void testSingeFace() throws IOException {
		FaceAnnotation[] faceAnnotations = mtcnnService.faceDetection("classpath:/Anthony_Hopkins_0002.jpg");
        String toEqual = "[{\"bbox\":{\"x\":74,\"y\":63,\"w\":99,\"h\":125},\"confidence\":0.9996323585510254," + 
            "\"landmarks\":[{\"type\":\"LEFT_EYE\",\"position\":{\"x\":101,\"y\":114}}," + 
            "{\"type\":\"RIGHT_EYE\",\"position\":{\"x\":148,\"y\":114}},"+
            "{\"type\":\"NOSE\",\"position\":{\"x\":124,\"y\":137}},"+
            "{\"type\":\"MOUTH_LEFT\",\"position\":{\"x\":105,\"y\":160}},"+
            "{\"type\":\"MOUTH_RIGHT\",\"position\":{\"x\":146,\"y\":160}}]}]";
		assertThat(toJson(faceAnnotations), equalTo(toEqual));
	}

	@Test
	public void testFailToDetectFace() throws IOException {
		FaceAnnotation[] faceAnnotations = mtcnnService.faceDetection("classpath:/broken.png");
		assertThat(toJson(faceAnnotations), equalTo("[]"));
	}

	@Test
	public void testMultiFaces() throws IOException {
		FaceAnnotation[] faceAnnotations = mtcnnService.faceDetection("classpath:/VikiMaxiAdi.jpg");
        String toEqual = "[{\"bbox\":{\"x\":332,\"y\":94,\"w\":57,\"h\":69},\"confidence\":0.9999037981033325,"+
            "\"landmarks\":[{\"type\":\"LEFT_EYE\","+
            "\"position\":{\"x\":346,\"y\":120}},"+
            "{\"type\":\"RIGHT_EYE\",\"position\":{\"x\":373,\"y\":121}},"+
            "{\"type\":\"NOSE\",\"position\":{\"x\":357,\"y\":134}},"+
            "{\"type\":\"MOUTH_LEFT\",\"position\":{\"x\":346,\"y\":147}},"+
            "{\"type\":\"MOUTH_RIGHT\",\"position\":{\"x\":370,\"y\":148}}]},"+
            "{\"bbox\":{\"x\":101,\"y\":155,\"w\":70,\"h\":83},\"confidence\":0.9995300769805908,"+
            "\"landmarks\":[{\"type\":\"LEFT_EYE\",\"position\":{\"x\":121,\"y\":188}},"+
            "{\"type\":\"RIGHT_EYE\",\"position\":{\"x\":153,\"y\":190}},"+
            "{\"type\":\"NOSE\",\"position\":{\"x\":136,\"y\":203}},"+
            "{\"type\":\"MOUTH_LEFT\",\"position\":{\"x\":122,\"y\":218}},"+
            "{\"type\":\"MOUTH_RIGHT\",\"position\":{\"x\":150,\"y\":220}}]},"+
            "{\"bbox\":{\"x\":102,\"y\":157,\"w\":66,\"h\":80},"+
            "\"confidence\":0.9988951086997986,"+
            "\"landmarks\":[{\"type\":\"LEFT_EYE\",\"position\":{\"x\":122,\"y\":188}},"+
            "{\"type\":\"RIGHT_EYE\",\"position\":{\"x\":153,\"y\":190}},"+
            "{\"type\":\"NOSE\",\"position\":{\"x\":136,\"y\":203}},"+
            "{\"type\":\"MOUTH_LEFT\",\"position\":{\"x\":122,\"y\":217}},"+
            "{\"type\":\"MOUTH_RIGHT\",\"position\":{\"x\":149,\"y\":219}}]},"+
            "{\"bbox\":{\"x\":332,\"y\":92,\"w\":56,\"h\":72},"+
            "\"confidence\":0.9974604845046997,"+
            "\"landmarks\":[{\"type\":\"LEFT_EYE\","+
            "\"position\":{\"x\":347,\"y\":121}},"+
            "{\"type\":\"RIGHT_EYE\",\"position\":{\"x\":374,\"y\":121}},"+
            "{\"type\":\"NOSE\",\"position\":{\"x\":358,\"y\":134}},"+
            "{\"type\":\"MOUTH_LEFT\",\"position\":{\"x\":348,\"y\":148}},"+
            "{\"type\":\"MOUTH_RIGHT\",\"position\":{\"x\":370,\"y\":148}}]}]";
	    assertThat(toJson(faceAnnotations), equalTo(toEqual));
	}

	private String toJson(FaceAnnotation[] faceAnnotations) throws JsonProcessingException {
		return new ObjectMapper().writeValueAsString(faceAnnotations);
	}
}
