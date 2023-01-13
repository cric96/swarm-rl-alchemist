package it.unibo.model

import it.unibo.model.network.NeuralNetworkEncoding
import it.unibo.model.network.torch.torch
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

case class State(distances: List[(Double, Double)], directionToLeader: (Double, Double))

object State {
  val neighborhood = 5 // fixed
  implicit val encoding = new NeuralNetworkEncoding[State] {
    override def elements: Int = (neighborhood + 1) * 2

    override def toSeq(elem: State): Seq[Double] =
      elem.distances.flatMap { case (l, r) =>
        List(l, r)
      } ++ List(elem.directionToLeader._1, elem.directionToLeader._2)

    override def toTensor(elem: State): py.Dynamic =
      torch.cat((torch.tensor(Seq(Seq(0.0, 0.0).toPythonCopy).toPythonCopy), torch.tensor(elem.distances.toPythonCopy)))
  }
}
