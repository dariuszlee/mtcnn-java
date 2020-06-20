package net.tzolov.cv.mtcnn;

import java.io.IOException;

import org.junit.Test;

public class MxNetLoaderTest {
	@Test
	public void testSingeFace() throws IOException {
        String modelPath = "/home/dzly/projects/countr_face_recognition/mtcnn-java/src/main/resources/mxnet_model/det1";
        MxNetLoader mxNetLoader = new MxNetLoader(new int[]{1, 3, 200, 200}, modelPath);    
        // mxNetLoader.Run();    
    }
}
