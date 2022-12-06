package it.unibo.model.network
import me.shadaj.scalapy.py

package object torch {
  val torch = py.module("torch")
  val nn = py.module("torch.nn")
  val optim = py.module("torch.optim")
  val log = py.module("torch.utils.tensorboard")
  val writer = log.SummaryWriter()
}
