package net.tzolov.cv.mtcnn;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;
import java.awt.image.BufferedImage;

import static org.nd4j.linalg.indexing.NDArrayIndex.all;
import static org.nd4j.linalg.indexing.NDArrayIndex.point;
import static org.nd4j.linalg.indexing.NDArrayIndex.interval;

import javax.imageio.ImageIO;

import org.apache.mxnet.infer.javaapi.Predictor;
import org.apache.mxnet.javaapi.Context;
import org.apache.mxnet.javaapi.DType;
import org.apache.mxnet.javaapi.DataDesc;
import org.apache.mxnet.javaapi.Layout;
import org.apache.mxnet.javaapi.NDArray;
import org.apache.mxnet.javaapi.Shape;
import org.datavec.image.loader.Java2DNativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;

public class MxNetLoader {
	private final Predictor proposeNetGraphRunner;
	private final Java2DNativeImageLoader imageLoader;
    private List<Context> ctx;
    // private static Shape inputShape = new Shape(new int[]{1, 3, 320, 327});
    private static Shape inputShape = new Shape(new int[]{1, 0, 3});


    public MxNetLoader(int[] scales){
		this.imageLoader = new Java2DNativeImageLoader();

        this.ctx = new ArrayList<>();
        this.ctx.add(Context.cpu()); // Choosing CPU Context here

        String modelPath = "/home/dzly/projects/countr_face_recognition/mtcnn-java/src/main/resources/mxnet_model/det1";
        List<DataDesc> inputDesc = new ArrayList<>();
        inputDesc.add(new DataDesc("data", new Shape(new int[]{1, 3, 25, 25}), DType.Float32(), Layout.NCHW()));
        proposeNetGraphRunner = new Predictor(modelPath, inputDesc, this.ctx, 0);
    }

    public List<INDArray> run(INDArray ndImage3HW){
        System.out.println("DARIUS " + ndImage3HW.shapeInfoToString());
        float[] ints = ndImage3HW.data().asFloat();
        Shape inputShape2  = new Shape(new int[]{1, 3, (int) ndImage3HW.shape()[2] , (int) ndImage3HW.size(3)});
        NDArray img = new NDArray(ints, inputShape2, this.ctx.get(0));
        img = img.reshape(new int[]{1,3, 25, 25});
        System.out.println("DARIUS" + img.shape());

        List<NDArray> imgs = new ArrayList<NDArray>();
        imgs.add(img);

        List<NDArray> outs = proposeNetGraphRunner.predictWithNDArray(imgs);
        System.out.println("DARIUS: " + outs.size());
        System.out.println("DARIUS: " + outs.get(0).shape());
        System.out.println("DARIUS: " + outs.get(1).shape());

        List<INDArray> outs2 = new ArrayList<>();
        outs2.add(ndImage3HW);
        return outs2;
    }

    public void Run() throws IOException {
        String imgPath = "/home/dzly/projects/countr_face_recognition/faceclient/src/test/resources/multi-face.jpg";

        try (InputStream imageInputStream = new FileInputStream(imgPath)) {
            BufferedImage inputImage = ImageIO.read(imageInputStream);
            System.out.println("DARIUS: "+ inputImage.getType());
            Java2DNativeImageLoader imageLoader = new Java2DNativeImageLoader();
            INDArray ndImage3HW = imageLoader.asMatrix(inputImage).get(point(0), interval(0, 3), all(), all());
            System.out.println("DARIUS " + ndImage3HW.shapeInfoToString());
            float[] ints = ndImage3HW.data().asFloat();
            Shape inputShape2  = new Shape(new int[]{1, 4, inputImage.getWidth(), inputImage.getHeight()});
            NDArray img = new NDArray(ints, inputShape2, this.ctx.get(0));
            img = img.reshape(new int[]{1,3, 25, 25});
            System.out.println("DARIUS" + img.shape());

            List<NDArray> imgs = new ArrayList<NDArray>();
            imgs.add(img);

            List<NDArray> outs = proposeNetGraphRunner.predictWithNDArray(imgs);
            System.out.println("DARIUS: " + outs.size());
            System.out.println("DARIUS: " + outs.get(0).shape());
            System.out.println("DARIUS: " + outs.get(1).shape());
        }
    }
}