package it.unibo.model.network

import me.shadaj.scalapy.py

trait NeuralNetworkEncoding[A] {
  def elements: Int
  def toSeq(elem: A): Seq[Double]
  def toTensor(elem: A): py.Dynamic
}
