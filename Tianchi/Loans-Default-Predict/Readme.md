​	天池零基础入门金融风控-贷款违约预测项目
https://tianchi.aliyun.com/competition/entrance/531830/information



+ 特征字段表分析

  一共46个字段，具体分类如下

  + ID字段：id
  + 数值型——31个
    + 实数型——10个
      loanAmnt，interestRate，installment，annualIncome，dti，ficoRangeLow，ficoRangeHigh，revolBal，revolUtil，totalAcc
    + 计数型——4+15=19个
      delinquency_2years，openAcc，pubRec，pubRecBankruptcies，n0至n14
    + 日期型——2个
      issueDate，earliesCreditLine
  + 类别型——10个
    + 有序型——5个
      term，grade，subGrade，employmentLength，homeOwnership
    + 离散型——5个
      verificationStatus，purpose，initialListStatus，applicationType，policyCode
  + 待定——4个
    employmentTitle，postCode，regionCode，title

|     **Field**      |                       **Description**                        | 存储类型和取值情况                                    | 实际类型   | 需要做的变换 |
| :----------------: | :----------------------------------------------------------: | ----------------------------------------------------- | ---------- | ------------ |
|         id         |                为贷款清单分配的唯一信用证标识                | int                                                   |            |              |
|      loanAmnt      |                           贷款金额                           | float                                                 | 实数型     |              |
|        term        |                       贷款期限（year）                       | int，只有3和5两种取值，                               | 有序型     |              |
|    interestRate    |                           贷款利率                           | float，取值较多，                                     | 实数型     |              |
|    installment     |                         分期付款金额                         | float，取值较多                                       | 实数型     |              |
|       grade        |                           贷款等级                           | str，取值A,B,C,D,E,F,G,                               | 有序型     |              |
|      subGrade      |                        贷款等级之子级                        | str，对应于上述等级，细分为A1,A2,A3,A4,A5             | 有序型     |              |
|  employmentTitle   |                           就业职称                           | float，取值较多，这个特征有点奇怪，按理应该是离散型的 | ？？       |              |
|  employmentLength  |                        就业年限（年）                        | str，取值为 <1 year，1 year 至 9 years，10+ years     | 有序型     |              |
|   homeOwnership    |              借款人在登记时提供的房屋所有权状况              | int，取值从 0 到 5，这个应当是序数型或者离散型        | 有序型     |              |
|    annualIncome    |                            年收入                            | float，取值很多                                       | 实数型     |              |
| verificationStatus |                           验证状态                           | int，取值 0 到 2                                      | 离散型     |              |
|     issueDate      |                        贷款发放的月份                        | str，形如2017-01-01的日期，不应当单独使用             | 区间型     |              |
|      purpose       |               借款人在贷款申请时的贷款用途类别               | int，从 0 到 13                                       | 离散型     |              |
|      postCode      |         借款人在贷款申请中提供的邮政编码的前3位数字          | int，取值较多，但是应当是离散型                       | ？？       |              |
|     regionCode     |                           地区编码                           | int，取值较多，应当是离散型                           | 离散型     |              |
|        dti         |                          债务收入比                          | float，取值较多                                       | 实数型     |              |
| delinquency_2years |       借款人过去2年信用档案中逾期30天以上的违约事件数        | int，取值不多，偏态分布                               | **计数型** |              |
|    ficoRangeLow    |            借款人在贷款发放时的fico所属的下限范围            | int，取值较多                                         | 实数型     |              |
|   ficoRangeHigh    |            借款人在贷款发放时的fico所属的上限范围            | int，取值较多                                         | 实数型     |              |
|      openAcc       |              借款人信用档案中未结信用额度的数量              | int，取值中等                                         | **计数型** |              |
|       pubRec       |                      贬损公共记录的数量                      | int，取值中等                                         | **计数型** |              |
| pubRecBankruptcies |                      公开记录清除的数量                      | int，取值 0 到 12                                     | **计数型** |              |
|      revolBal      |                       信贷周转余额合计                       | float，                                               | 实数型     |              |
|     revolUtil      | 循环额度利用率，或借款人使用的相对于所有可用循环信贷的信贷金额 | int，取值较多，[0, 100] ，                            | 实数型     |              |
|      totalAcc      |              借款人信用档案中当前的信用额度总数              | int，                                                 | 实数型     |              |
| initialListStatus  |                      贷款的初始列表状态                      | int，取值 0 和 1                                      | 离散型     |              |
|  applicationType   |       表明贷款是个人申请还是与两个共同借款人的联合申请       | int，取值 0 和 1                                      | 离散型     |              |
| earliesCreditLine  |              借款人最早报告的信用额度开立的月份              | str，Aug-2001之类的格式                               | 区间型     |              |
|       title        |                     借款人提供的贷款名称                     | int，取值非常多，应当是离散型                         | ？？       |              |
|     policyCode     |      公开可用的策略_代码=1新产品不公开可用的策略_代码=2      | int，理论取值 1或2，实际只有1                         | 离散型     |              |
|   n系列匿名特征    |        匿名特征n0-n14，为一些贷款人行为计数特征的处理        | int，取值不多                                         | **计数型** |              |