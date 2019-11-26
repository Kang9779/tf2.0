# tf2.0
#### 1.常用函数解释
> tf.matmul(x,y)
- 实现两个矩阵x和y相乘

> tf.squee(input,axis=None,name=None)
- 对input矩阵的shape为1的维度进行压缩，保持数据不变
    * 例如：shape(2,1) -> shape(2,)<br>shape(2,1,3)-> shape(2,3)

> tf.nn.sparse_softmax_cross_entropy_with_logits(labels=None,logits=None,name=None)

- 对logits先计算softmax,然后计算cross_entropy
    - loss = tf.reduce_mean(tf.nn.sparse_cross_entropy_with_logits(labels=labels,logits=logits))
        - 其中labels标签为真实的标签，不是one-hot后的标签
    - tf.nn.softmax_cross_entropy_with_logits() 其中的labels是one-hot后的标签
> tf.cast(input,dtype=None)
- 实现input的类型转换
    - 例如：x = tf.cast(x,dtype=tf.float32)





