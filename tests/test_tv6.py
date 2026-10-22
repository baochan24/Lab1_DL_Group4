"""TV6 boundary tests; no FashionMNIST results or actual torch reload claims."""
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from src import utils
from src.predict import validate_checkpoint_metadata


def config():
    return json.loads((utils.ROOT / 'configs/A.json').read_text())


def checkpoint():
    # Schema-only fixture; not a serialized model or result of a real run.
    return {'model_state_dict': {'weight': object()},
            'model_config': {'model_name': 'mlp', 'hidden_units': 128},
            'preprocessing': {'transform': 'to_tensor', 'data_root': 'data'},
            'class_names': [f'class_{i}' for i in range(10)],
            'run_id': 'fixture', 'epoch': 1, 'val_metrics': {'loss': 0.5, 'accuracy': 0.5}}


class ContractTests(unittest.TestCase):
    def test_config_rejects_path_escape(self):
        with self.assertRaises(ValueError):
            utils.validate_config({**config(), 'run_id': '../A'})

    def test_json_rejects_nan_without_destroying_existing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'metadata.json'
            utils.save_json(path, {'status': 'existing'})
            with self.assertRaises(ValueError):
                utils.save_json(path, {'loss': float('nan')})
            self.assertEqual(utils.load_json(path), {'status': 'existing'})

    def test_split_rejects_duplicate_and_out_of_range(self):
        cfg = config()
        split = {'dataset_size': 60000, 'seed': 42,
                 'train_indices': list(range(54000)), 'val_indices': list(range(54000, 60000))}
        utils.validate_split(split, cfg)
        for bad in (0, 60000, True):
            changed = copy.deepcopy(split)
            changed['val_indices'][0] = bad
            with self.assertRaises(ValueError):
                utils.validate_split(changed, cfg)

    def test_split_rejects_wrong_seed(self):
        with self.assertRaises(ValueError):
            utils.validate_split({'dataset_size': 60000, 'seed': 0}, config())

    def test_checkpoint_valid_schema(self):
        validate_checkpoint_metadata(checkpoint())

    def test_checkpoint_rejects_percent_accuracy(self):
        value = checkpoint()
        value['val_metrics']['accuracy'] = 85
        with self.assertRaises(ValueError):
            validate_checkpoint_metadata(value)

    def test_checkpoint_rejects_nonfinite_loss(self):
        value = checkpoint()
        value['val_metrics']['loss'] = float('nan')
        with self.assertRaises(ValueError):
            validate_checkpoint_metadata(value)

    def test_checkpoint_rejects_missing_preprocessing(self):
        value = checkpoint()
        del value['preprocessing']['data_root']
        with self.assertRaises(ValueError):
            validate_checkpoint_metadata(value)

    def test_checkpoint_rejects_duplicate_class_names(self):
        value = checkpoint()
        value['class_names'] = ['same'] * 10
        with self.assertRaises(ValueError):
            validate_checkpoint_metadata(value)

    def test_checkpoint_rejects_invalid_epoch_or_model(self):
        for field, bad in [('epoch', 0), ('model_config', {'model_name': 'mlp', 'hidden_units': True}),
                           ('val_metrics', None)]:
            value = checkpoint()
            value[field] = bad
            with self.assertRaises(ValueError):
                validate_checkpoint_metadata(value)


class OrchestrationTests(unittest.TestCase):
    """Use temporary outputs and substitute member functions at their public boundary."""
    def setUp(self):
        self.main = importlib.import_module('main')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = {**config(), 'output_root': str(self.root / 'outputs'),
                    'split_indices_path': str(self.root / 'split.json')}
        utils.save_json(self.root / 'config.json', self.cfg)
        utils.save_json(self.root / 'split.json', {})

    def run_train(self, fit, plot):
        class Model:
            def __init__(self, **kwargs):
                pass
            def to(self, device):
                return self
        modules = {}
        for name, attributes in {
            'src.data': {'build_dataloaders': lambda cfg: ([], [], [], [str(i) for i in range(10)])},
            'src.model': {'MLP': Model},
            'src.train': {'fit': fit},
            'src.visualize': {'plot_history': plot},
        }.items():
            module = types.ModuleType(name)
            module.__dict__.update(attributes)
            modules[name] = module
        with patch.dict(sys.modules, modules), \
             patch.object(utils, 'set_seed'), patch.object(utils, 'resolve_device', return_value='cpu'), \
             patch.object(utils, 'validate_split'), \
             patch.object(sys, 'argv', ['main.py', '--mode', 'train', '--config', str(self.root/'config.json')]):
            self.main.main()

    def metadata(self):
        return utils.load_json(self.root / 'outputs/runs/A/metadata.json')

    def test_failed_training_has_failed_status(self):
        def fail(*args):
            raise RuntimeError('member failure fixture')
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            self.run_train(fail, lambda *args: None)
        self.assertEqual(self.metadata()['status'], 'failed')
        self.assertGreaterEqual(self.metadata()['duration_seconds'], 0)

    def test_plot_failure_preserves_successful_training_status(self):
        def fit(model, train, val, cfg, folder):
            # Temporary sentinel files only; never packaged as model outputs.
            history, cp = folder/'history.csv', folder/'best_model.pth'
            history.write_text('fixture')
            cp.write_bytes(b'fixture')
            return {'history_path': str(history), 'best_checkpoint_path': str(cp), 'best_epoch': 1}
        def plot(*args):
            raise RuntimeError('plot fixture')
        with self.assertRaisesRegex(RuntimeError, 'plot fixture'):
            self.run_train(fit, plot)
        self.assertEqual(self.metadata()['status'], 'trained')
        self.assertEqual(self.metadata()['plot_status'], 'failed')

    def test_successful_orchestration_and_no_overwrite(self):
        def fit(model, train, val, cfg, folder):
            (folder/'history.csv').write_text('fixture')
            (folder/'best_model.pth').write_bytes(b'fixture')
            return {'history_path': str(folder/'history.csv'),
                    'best_checkpoint_path': str(folder/'best_model.pth'), 'best_epoch': 1}
        self.run_train(fit, lambda *args: None)
        self.assertEqual(self.metadata()['status'], 'completed')
        with self.assertRaises(FileExistsError):
            self.run_train(fit, lambda *args: None)


if __name__ == '__main__':
    unittest.main()
