
import org.apache.spark.{SparkContext, SparkConf}

object SparkWordCount {

  def main(args: Array[String]) {

    System.setProperty("hadoop.home.dir","c:\\winutils");

    val sparkConf = new SparkConf().setAppName("SparkWordCount").setMaster("local[*]")

    val sc=new SparkContext(sparkConf)

    val input=sc.textFile("input")

    val wc=input.flatMap(line=>{line.split(" ")}).map(word=>(word.charAt(0),word+" ")).cache()

    val output=wc.reduceByKey(_+_)

    output.saveAsTextFile("output")



  }

}
