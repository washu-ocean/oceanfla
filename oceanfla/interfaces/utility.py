from nipype.interfaces.utility.base import MergeInputSpec, _ravel
from nipype.interfaces.io import IOBase, add_traits
from nipype.interfaces.base import (
    BaseInterfaceInputSpec,
    SimpleInterface,
    TraitedSpec,
    DynamicTraitedSpec,
    traits,
    CommandLine
)
from bids.utils import listify
from niworkflows.interfaces.bids import DerivativesDataSink, _DerivativesDataSinkInputSpec
from nipype.interfaces.base.support import RuntimeContext, InterfaceResult
from nipype import config
from nipype.utils.filemanip import indirectory
import os


def optional_trait(*in_traits, **kwargs):
    return traits.Union(
        None,
        *in_traits,
        **kwargs
    )


class MergeUnique(IOBase):
    input_spec = MergeInputSpec
    output_spec = DynamicTraitedSpec
    _sep = "_x"

    def __init__(self, collapse_none=True, **inputs):
        super().__init__(**inputs)
        self._collapse_none = collapse_none

    # def _run_interface(self, runtime):
    def _list_outputs(self):
        outputs = self._outputs().get()
        # return super()._list_outputs()
        input_keys = [k.split(self._sep) for k in self.inputs.get().keys()]
        input_keys = [t for t in input_keys if len(t) == 2 and t[1].isnumeric()]
        input_key_map = dict()
        for key, num in input_keys:
            if key in input_key_map:
                input_key_map[key].append(num)
            else:
                input_key_map[key] = [num]

        if len(input_key_map) < 1:
            return outputs

        for key, num_list in input_key_map.items():
            out = []
            values = [getattr(self.inputs, f"{key}{self._sep}{num}")
                      for num in sorted(num_list, key=lambda x: int(x))
                      if hasattr(self.inputs, f"{key}{self._sep}{num}")]
            if self.inputs.axis == "vstack":
                for value in values:
                    if isinstance(value, list) and not self.inputs.no_flatten:
                        out.extend(
                            _ravel(value) if self.inputs.ravel_inputs else value)
                    else:
                        out.append(value)
            else:
                lists = [listify(val) if val is not None else [None]
                         for val in values]
                out = [[val[i] for val in lists] for i in range(len(lists[0]))]

            if all([o is None for o in out]) and self._collapse_none:
                out = None
            outputs[key] = out
        return outputs


class ExtractDataGroupInputSpec(DynamicTraitedSpec, BaseInterfaceInputSpec):
    task = traits.Str(desc="The task name of the data")

    event_idx = traits.Int(desc="The index of the event task name")

    event_tasks = traits.List(traits.Str, desc="List of task names pertaining to event files")

    run = traits.Str(desc="The run number of the data")


class ExtractDataGroupOutputSpec(TraitedSpec):
    bold_file = traits.File(
        exists=True,
        desc="The described bold file"
    )
    confounds_file = traits.File(
        exists=True,
        desc="The described confounds file"
    )
    events_file = traits.File(
        exists=True,
        desc="The described events file"
    )


class ExtractDataGroup(IOBase):
    input_spec = ExtractDataGroupInputSpec
    output_spec = DynamicTraitedSpec

    def _list_outputs(self):
        outputs = self._outputs().get()
        if self.inputs.event_tasks:
            event_task_needed = self.input.event_tasks[self.inputs.event_idx]
        else:
            event_task_needed = self.inputs.task
        for input_name in self.inputs.get().keys():
            if input_name in ["task", "run", "event_tasks", "event_idx"]:
                continue
            if getattr(self.inputs, input_name) is None:
                outputs[input_name] = None
            else:
                outputs[input_name] = extract_task_run_file(
                    bids_list=getattr(self.inputs, input_name),
                    task_needed=self.inputs.task,
                    event_task_needed=event_task_needed,
                    run_needed=self.inputs.run
                )
        return outputs


def extract_task_run_file(bids_list: list,
                          task_needed: str,
                          event_task_needed: str | None,
                          run_needed: str):
    from bids.layout import parse_file_entities
    from pathlib import Path

    for file in bids_list:
        fpath = Path(file)
        parse_path = Path(fpath.parent.name) / fpath.name
        bids_file_entities = parse_file_entities(str(parse_path))
        run = int(bids_file_entities.get("run", 1))
        if run == int(run_needed):
            if bids_file_entities["suffix"] == "events" and bids_file_entities["task"] == event_task_needed:
                return file
            elif bids_file_entities["suffix"] != "events" and bids_file_entities["task"] == task_needed:
                return file
    raise RuntimeError(
        f"Could not find a file with entities task-{task_needed[0]} or task-{task_needed[1]}, run-{run_needed}")


class ReadMetadataFileInputSpec(BaseInterfaceInputSpec):
    bids_file = traits.File(
        exists=True,
        desc="A BIDS image/data file"
    )


class ReadMetadataFileOutputSpec(DynamicTraitedSpec):
    metadata_dict = traits.Dict(
        key_trait=traits.Str
    )


class ReadMetadataFile(SimpleInterface):
    input_spec = ReadMetadataFileInputSpec
    output_spec = ReadMetadataFileOutputSpec

    def __init__(self, fields=None, error_on_missing=False, **inputs):
        from bids.utils import listify

        super().__init__(**inputs)
        self._fields = listify(fields or [])
        self._error_on_missing = error_on_missing

    def _outputs(self):
        base = super()._outputs()
        if self._fields:
            base = add_traits(base, self._fields)
        return base

    def _run_interface(self, runtime):

        metadata_results = read_metadata_file(
            data_file=self.inputs.bids_file,
            requested_fields=self._fields,
            strict=self._error_on_missing
        )
        self._results.update(metadata_results)

        return runtime


def read_metadata_file(data_file: str,
                       requested_fields: list[str],
                       strict: bool = True):
    from oceanfla.utilities import replace_entity
    from pathlib import Path
    import json

    res_dict = {}
    metadata_dict = {}
    metadata_path = Path(replace_entity(data_file, "ext", ".json"))

    if metadata_path.exists():
        with open(metadata_path, "r") as metaf:
            metadata_dict = json.load(metaf)

        for fname in requested_fields:
            if strict and fname not in metadata_dict:
                raise KeyError(
                    f'Metadata field "{fname}" not found for file {data_file}'
                )
            res_dict[fname] = metadata_dict.get(fname, traits.Undefined)

    res_dict["metadata_dict"] = metadata_dict
    return res_dict


class OptionalInterfaceSpec(DynamicTraitedSpec):
    execute = traits.Bool(default_value=True, usedefault=True)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        inital_trait_values = self.trait_get()
        for trait_name in self.copyable_trait_names():
            current_trait = self.traits()[trait_name]
            if hasattr(current_trait, "name_source"):
                continue
            # if self.remove_trait(trait_name):
            self.add_trait(
                trait_name,
                traits.Union(
                    current_trait,
                    None,
                    # default_value=None,
                    usedefualt=current_trait.usedefault
                )
                # [current_trait, None],
            )
            # trait_set_value = inital_trait_values[trait_name] if inital_trait_values[trait_name] else traits.Undefined
            # if current_trait.usedefault:
            #     trait_set_value = current_trait.default_value()[1]
            # if trait_name == "execute":
            #     trait_set_value = True
            self.trait_set(**{trait_name: inital_trait_values[trait_name]})


class OptionalInterface(SimpleInterface):
    def run(self, cwd=None, ignore_exception=None, **inputs):
        """Execute this interface.

        This interface will not raise an exception if runtime.returncode is
        non-zero.

        Parameters
        ----------
        cwd : specify a folder where the interface should be run
        inputs : allows the interface settings to be updated

        Returns
        -------
        results :  :obj:`nipype.interfaces.base.support.InterfaceResult`
            A copy of the instance that was executed, provenance information and,
            if successful, results

        """
        # print("IN RUN")
        if hasattr(self.inputs, "execute") and not getattr(self.inputs, "execute"):
            rtc = RuntimeContext(
                resource_monitor=config.resource_monitor and self.resource_monitor,
                ignore_exception=(
                    ignore_exception
                    if ignore_exception is not None
                    else self.ignore_exception
                ),
            )

            with indirectory(cwd or os.getcwd()):
                self.inputs.trait_set(**inputs)

            with rtc(self, cwd=cwd, redirect_x=self._redirect_x) as runtime:
                outputs = None
                inputs = self.inputs.get_traitsfree()
                # SKIP Run interface
                # runtime = self._pre_run_hook(runtime)
                # runtime = self._run_interface(runtime)
                # runtime = self._post_run_hook(runtime)
                # Collect outputs
                outputs = self._outputs()
                if hasattr(outputs, "execute"):
                    setattr(outputs, "execute", False)

            results = InterfaceResult(
                self.__class__,
                rtc.runtime,
                inputs=inputs,
                outputs=outputs,
                provenance=None,
            )
            return results
        else:
            return super().run(cwd=cwd, ignore_exception=ignore_exception, **inputs)


class OptionalCommandLineInterface(CommandLine):
    def run(self, cwd=None, ignore_exception=None, **inputs):
        """Execute this interface.

        This interface will not raise an exception if runtime.returncode is
        non-zero.

        Parameters
        ----------
        cwd : specify a folder where the interface should be run
        inputs : allows the interface settings to be updated

        Returns
        -------
        results :  :obj:`nipype.interfaces.base.support.InterfaceResult`
            A copy of the instance that was executed, provenance information and,
            if successful, results

        """
        # print("IN RUN")
        if hasattr(self.inputs, "execute") and not getattr(self.inputs, "execute"):
            rtc = RuntimeContext(
                resource_monitor=config.resource_monitor and self.resource_monitor,
                ignore_exception=(
                    ignore_exception
                    if ignore_exception is not None
                    else self.ignore_exception
                ),
            )

            with indirectory(cwd or os.getcwd()):
                self.inputs.trait_set(**inputs)

            with rtc(self, cwd=cwd, redirect_x=self._redirect_x) as runtime:
                outputs = None
                inputs = self.inputs.get_traitsfree()
                # SKIP Run interface
                # runtime = self._pre_run_hook(runtime)
                # runtime = self._run_interface(runtime)
                # runtime = self._post_run_hook(runtime)
                # Collect outputs
                outputs = self._outputs()
                if hasattr(outputs, "execute"):
                    setattr(outputs, "execute", False)

            results = InterfaceResult(
                self.__class__,
                rtc.runtime,
                inputs=inputs,
                outputs=outputs,
                provenance=None,
            )
            return results
        else:
            return super().run(cwd=cwd, ignore_exception=ignore_exception, **inputs)


class FLADataSinkInputSpec(_DerivativesDataSinkInputSpec, OptionalInterfaceSpec):
    in_file = traits.Union(
        traits.File(exists=True),
        traits.List(),
        None,
        desc="the object to be saved"
    )


class FLADataSink(DerivativesDataSink, OptionalInterface):
    input_spec = FLADataSinkInputSpec

    def __init__(self, allowed_entities=None, out_path_base=None, extra_bids_patterns=None, **inputs):
        super().__init__(allowed_entities=allowed_entities, out_path_base=out_path_base, **inputs)
        self._file_patterns += tuple(extra_bids_patterns)
