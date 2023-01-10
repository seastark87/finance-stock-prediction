import torch.nn as nn
from pytorch_lightning import Trainer, seed_everything
from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import TensorBoardLogger
from datamodule import *
from classifier import ClassifyCodes
from model import *

if __name__ == '__main__':
    stocks = ['095570', '006840', '027410', '282330', '138930', '001460', '001040', '079160', '00104K', '000120', '011150', '011155',
              '001045', '097950', '097955', '000590', '012030', '016610', '005830', '000990', '000995', '139130', '001530', '000210',
              '001880', '000215', '375500', '37550K', '004840', '155660', '069730', '017940', '365550', '114090', '078930', '006360',
              '001250', '007070', '078935', '012630',  '089470', '294870', '011200', '082740', '003560', '175330', '234080', '001060',
              '001067', '001065', '096760', '105560', '002380', '344820', '009070', '009440', '119650', '092220', '016380', '016385',
              '001390', '033180', '001940', '025000', '092230', '000040', '044450', '030200', '033780', '030210', '058850', '058860',
              '093050',  '034220', '051900', '051905', '003555', '032640', '011070', '066570', '066575', '037560', '051910', '051915',
              '079550', '006260', '010120', '000680', '229640', '001120', '108670', '108675',  '023150', '035420', '181710', '005940',
              '005945', '338100', '034310', '008260', '004250', '004255', '010060', '005490', '010950', '010955', '034120', '101060',
              '005090', '001380', '004060', '002360', '009160', '123700', '025530', '034730', '011790', '018670', '001740', '001745',
              '006120', '006125', '210980', '068400', '302440', '326030',  '03473K', '096770', '096775', '001510', '001515', '285130',
              '28513K', '017670', '000660', '064960', '100840', '003570', '036530', '005610', '011810', '077970', '071970', '002820',
              '084870', '002710', '002900', '024070', '037270', '000500', '000860', '035250', '011420', '002100', '009450', '267290',
              '012320', '000050', '214390', '012610', '009140', '013580', '012200', '012205', '002140', '010130', '002240', '009290',
              '017040', '017900', '037710', '030610', '339770', '007690', '005320', '001140', '002720', '083420', '014530', '014280',
              '014285', '008870', '001570', '002990', '002995', '011780', '011785', '214330', '001210', '073240', '092440', '000270',
              '024110', '013700', '004540', '004545', '001260', '008350', '008355', '004270', '003920', '003925', '025860', '005720',
              '005725', '002350', '002355', '003580', '251270', '090350', '090355', '000320', '000325', '006280', '005250', '005257',
              '004370', '072710', '058730', '023590', '019680', '019685', '006370', '008060', '00806K', '353200', '35320K', '000490',
              '008110', '005750', '006570', '001680', '001685', '084690', '084695', '128820', '117580', '016710', '003540', '003547',
              '003545', '009190', '014160', '047040', '009320', '042660', '003090', '069620', '000430', '006340', '006345', '003220',
              '024890', '002880', '000300', '012800', '015230', '001070', '006650', '001440', '084010', '001790', '001795', '001130',
              '003490', '003495', '005880', '003830', '016090', '069460', '192080', '012510', '004830', '004835', '024900', '145720',
              '002150', '001230', '023450', '004140', '007590', '005960', '005965', '026960', '002210', '102260', '000640', '170900',
              '028100', '282690', '001520', '001527', '001529', '084670', '082640', '001525', '008970', '092780', '049770', '018500',
              '006040', '030720', '014820', '014825', '163560', '004890', '002690', '000020', '000150', '000157', '241560', '000155',
              '034020', '336260', '33626K', '33626L', '016740', '192650', '024090', '003160', '092200', '013570', '210540', '007340',
              '026890', '115390', '032350', '330590', '000400', '023530', '004000', '286940', '280360', '004990', '00499K', '005300',
              '005305', '011170', '002270', '071840', '027740', '204320', '001080', '088980', '094800', '138040', '008560', '000060',
              '090370', '017180', '009900', '012690', '005360', '204210', '009680', '009580', '009200', '033920', '008420', '025560',
              '007120', '357250', '085620', '006800', '00680K', '006805', '002840', '268280', '107590', '134380', '003650', '155900',
              '003610', '001340', '035150', '002410', '096300', '007210', '002760', '003850', '000890', '003000', '001270', '001275',
              '026940', '015350', '011390', '005030', '002070', '100220', '030790', '005180', '003960', '008040', '007160', '014710',
              '006090', '001470', '006400', '006405', '006660', '028260', '02826K', '207940', '032830', '018260', '028050', '009150',
              '009155', '005930', '005935', '001360', '010140', '010145', '016360', '068290', '029780', '000810', '000815', '006110',
              '145990', '145995', '003230', '002170', '272550', '000070', '000075', '002810', '005680', '003720', '023000', '004380',
              '002450', '004440', '000520', '009770', '005500', '004690', '010960', '004450', '009470', '011230', '001820', '000390',
              '001290', '041650', '075180', '007540', '248170', '007860', '200880', '017390', '004410', '004415', '021050', '008490',
              '007610', '136490', '014910', '014915', '003080', '004980', '004985',  '000180',  '004360', '004365', '004490', '001430',
              '306200', '003030', '019440', '058650',  '091090', '067830', '033530', '075580', '027970', '145210', '308170', '068270',
              '336370', '33637K', '33637L', '248070', '004430', '017550', '053210', '134790', '016590', '029530', '004970', '011930',
              '005390', '004170', '035510', '034300', '031430', '031440', '006880', '005800', '001720', '001725', '009270', '009275',
              '002700', '019170', '019175', '002870', '293940', '055550', '001770', '004080', '102280', '003410',  '004770',  '004920',
              '112610', '008700', '002790', '00279K', '002795', '090430', '090435', '002030', '183190', '002310', '012170',   '122900',
              '010780', '001780', '018250', '161000', '011090', '005850', '012750', '023960',  '140910', '078520', '015260', '007460',
              '003060', '244920', '036570', '138250', '085310', '009810', '900140', '097520', '014440', '111770', '009970', '003520',
              '000670', '006740', '012280', '012160', '015360', '007310',  '271560', '001800', '070960', '316140', '033660', '118000',
              '010050', '006980', '017370', '105840', '010400', '049800', '016880', '095720', '005820', '010600', '008600', '033270',
              '014830', '000910', '047400', '011330', '077500', '002920', '000700', '003470', '003475',  '072130', '000220', '000225',
              '000227', '001200', '000100', '000105', '003460', '003465', '008730', '008250', '025820', '214320', '088260', '139480',
              '007660', '005950', '015020', '093230', '074610', '102460', '084680', '350520', '334890', '000760', '014990', '101140',
              '006490', '023800', '034590', '129260', '023810', '249420', '000230', '013360', '003120', '003200', '007110', '007570',
              '007575', '008500', '081000', '020760', '020150', '103590', '015860', '226320', '317400', '033240', '000950', '348950',
              '194370', '025620', '036420', '030000', '271980', '001560', '002620', '006220', '089590', '004910', '004700', '001550',
              '000480', '120030', '018470', '002600', '185750', '063160', '001630', '044380', '013890', '013870', '071320', '035000',
              '088790', '003780', '010640', '100250', '051630', '272450', '011000', '002780', '002787', '002785', '009310', '000650',
              '033250', '035720', '006380',  '001620', '029460', '281820', '145270', '357120', '007815', '007810', '00781K', '003690',
              '192820', '044820', '005070', '005420', '071950', '002020', '003070', '003075', '144620', '002025', '120110', '120115',
              '138490', '021240', '031820', '192400', '284740', '015590', '264900', '26490K', '005740', '005745', '020120', '039490',
              '014580', '015890', '006890', '003240', '011280', '004100', '004105', '009410', '009415', '001420', '007980', '055490',
              '078000', '214420', '019180', '363280', '36328K', '091810', '004870', '005690', '036580', '004720', '028670', '010820',
              '016800', '001020', '090080', '010770', '058430', '047050', '003670',  '017810', '103140', '005810', '950210', '086790',
              '293480', '039130', '172580', '153360', '352820', '071090', '019490', '000080', '000087', '000140', '000145', '152550',
              '036460', '005430', '071050', '071055', '010040', '025540', '004090', '002200', '002960', '000240', '123890', '015760',
              '006200', '009540', '023350', '025890', '000970', '104700', '017960', '161890', '024720', '161390', '034830', '007280',
              '168490', '010100', '047810', '123690', '003350', '011500', '002390', '014790', '060980', '053690', '042700', '008930',
              '128940', '009240', '020000', '003680', '105630', '069640', '016450', '010420', '009180', '213500', '014680', '004710',
              '004150', '025750', '004960', '011700', '001750', '001755', '018880', '009420', '014130', '300720', '002220', '006390',
              '003300', '051600', '052690', '130660', '002320', '097230', '003480', '180640', '18064K', '005110', '009460', '000880',
              '00088K', '088350', '000370', '009830', '009835', '272210', '012450', '000885', '003530', '003535', '195870', '101530',
              '143210', '000720', '267270', '000725', '005440', '086280', '064350', '079430', '012330', '010620', '069960', '004560',
              '004565', '004310', '322000', '017800', '307950', '011210', '267260', '004020', '267250', '005380', '005387', '005389',
              '005385', '001500', '011760', '227840', '126560', '001450', '057050', '093240', '003010', '111110', '008770', '008775',
              '002460', '378850', '241590', '006060', '013520', '010690', '133820', '010660', '000850', '016580', '032560', '004800',
              '094280', '298040', '298050', '298020', '298000', '093370', '081660', '005870', '079980', '005010', '069260', '000540',
              '000547', '000545']

    # define parameters
    params = dict(seq_len=30,
                  batch_size=256,
                  MSE=nn.MSELoss(),
                  max_epochs=50,
                  features=7,
                  hiddens=256,
                  num_layers=2,
                  dropout=0.2,
                  learning_rate=1.0e-4)

    # eliminate randomness
    seed_everything(42)

    # datamodule
    dm = FATDataModule(clf_stock=ClassifyCodes(stocks, N=1, option=1),
                       seq_len=params['seq_len'],
                       batch_size=params['batch_size'])

    # trainer
    logger = TensorBoardLogger('tb_logs', name='FAT_HC')

    trainer = Trainer(max_epochs=params['max_epochs'],
                      logger=logger,
                      gpus=0)

    # model
    model = lstm_model(features=params['features'],
                       hiddens=params['hiddens'],
                       seq_len=params['seq_len'],
                       batch_size=params['batch_size'],
                       MSE=params['MSE'],
                       num_layers=params['num_layers'],
                       dropout=params['dropout'],
                       learning_rate=params['learning_rate'])

    # fit and test
    trainer.fit(model, dm)
    trainer.test(model, datamodule=dm)
